#!/usr/bin/env python3
# filepath: auto_labeler.py

import gc
import os
import cv2
import numpy as np
import torch
import argparse
import time
import json
import random
import warnings
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from contextlib import contextmanager
import torchvision.transforms as T
from PyQt6.QtGui import QColor
import torch.nn.functional as F
import torchmetrics
from PIL import Image

from segmenter_sam2 import Segmenter
from point_selection_strategies import PointSelectionFactory
from plas.segmenter_plas import SuperpixelLabelExpander

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
# Suppress SAM2 optional post-processing extension warning
import warnings as _w
_w.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*cannot import name '_C' from 'sam2'.*"
)

# Disable PyTorch SDPA flash/mem-efficient kernels and silence related chatter
try:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

# Silence SDPA kernel selection messages and dtype hints
_w.filterwarnings("ignore", category=UserWarning, message=r".*Flash Attention kernel failed due to:.*")
_w.filterwarnings("ignore", category=UserWarning, message=r".*Falling back to all available kernels for scaled_dot_product_attention.*")
_w.filterwarnings("ignore", category=UserWarning, message=r".*Expected query, key and value to all be of dtype.*scaled_dot_product_attention.*")

# Optional multi-scale helpers (non-fatal if missing)
try:
    from multi_scale_utils import (
        choose_scale, ScaleConfig, downscale_image,
        init_coverage_map, update_coverage_map
    )
except Exception:  # pragma: no cover - graceful fallback
    choose_scale = None
    ScaleConfig = None
    downscale_image = None
    init_coverage_map = None
    update_coverage_map = None

@contextmanager
def timer(name=None):
    """Context manager for timing operations."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        print(f"{name}: {elapsed:.3f}s")

class PipelineTimer:
    """Precise timing tracker for different segmentation pipelines."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all timers for a new image."""
        self.times = {
            'setup': 0.0,
            'point_selection': 0.0,
            'sam2_propagation': 0.0,
            'mask_merging': 0.0,
            'plas_expansion': 0.0,
            'postprocessing': 0.0,
            'io_operations': 0.0
        }
        self.total_start = None
    
    def start_total(self):
        """Start timing the total processing for this image."""
        self.total_start = time.time()
    
    def add_time(self, operation, duration):
        """Add time for a specific operation."""
        if operation in self.times:
            self.times[operation] += duration
    
    @contextmanager
    def time_operation(self, operation):
        """Context manager to time a specific operation."""
        start = time.time()
        yield
        duration = time.time() - start
        self.add_time(operation, duration)
    
    def get_pipeline_times(self):
        """Calculate times for different pipeline combinations."""
        if self.total_start is None:
            total_time = sum(self.times.values())
        else:
            total_time = time.time() - self.total_start
        
        # Pure times (no overlap)
        sam2_only_time = (
            self.times['setup'] + 
            self.times['point_selection'] + 
            self.times['sam2_propagation'] + 
            self.times['mask_merging'] +
            self.times['postprocessing']
        )
        
        plas_only_time = (
            self.times['setup'] + 
            self.times['point_selection'] + 
            self.times['plas_expansion'] +
            self.times['postprocessing']
        )
        
        combined_time = (
            self.times['setup'] + 
            self.times['point_selection'] + 
            self.times['sam2_propagation'] + 
            self.times['mask_merging'] +
            self.times['plas_expansion'] +
            self.times['postprocessing']
        )
        
        return {
            'total_measured': total_time,
            'sam2_propagation_pipeline': sam2_only_time,
            'plas_pipeline': plas_only_time,
            'combined_pipeline': combined_time,
            'breakdown': self.times.copy()
        }

class AutoLabeler:
    def __init__(self,
                 images_dir,
                 ground_truth_dir=None,  # Optional
                 output_dir=None,
                 sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
                 sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
                 save_visualizations=False,
                 debug_save_expanded_masks=False,
                 device="cuda",
                 point_selection_strategy="random",
                 num_points=30,
                 use_maskSLIC=False,
                 num_classes=None,
                 downscale_auto=False,
                 downscale_fixed=None,
                 seed=None,
                 **strategy_kwargs):
        """Initialize the unified AutoLabeler with paths and parameters."""

        # Background semantics
        self.DEFAULT_BACKGROUND_CLASS_ID = 34
        self.DEFAULT_BACKGROUND_COLOR = (63, 69, 131)  # RGB

        # Paths & config
        self.images_dir = Path(images_dir)
        self.ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config

        # Device selection
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device
        # Store seed for reproducibility
        self.seed = seed

        self.save_visualizations = save_visualizations
        # Debug option: save every expanded mask separately + overlap counts
        self.save_expanded_masks_debug = debug_save_expanded_masks
        self.num_classes = num_classes  # May be determined later
        self.eval_mask_type = "propagation_plas"

        # Strategy configuration
        self.point_selection_strategy = point_selection_strategy
        self.num_points = num_points
        self.use_maskSLIC = use_maskSLIC
        self.strategy_kwargs = strategy_kwargs

        # Instantiate strategy
        self.strategy = PointSelectionFactory.create_strategy(
            self.point_selection_strategy,
            num_points=self.num_points,
            **self.strategy_kwargs,
        )
        # GT point sampling configuration (max points per GT mask used for overlap resolution)
        # Can be overridden via strategy kwargs: max_gt_points_per_mask
        self.max_gt_points_per_mask = self.strategy_kwargs.get('max_gt_points_per_mask', 3000)
        self.min_gt_points_full_mask = self.strategy_kwargs.get('min_gt_points_full_mask', 2500)
        # Cached listings
        self._cached_image_files = None

        # Interactive strategies (exclude SAM2_guided)
        self.is_interactive_strategy = self.point_selection_strategy in [
            "dynamicPoints_onlyA",
            "dynamicPoints",
            "dynamicPointsLargestGT",
        ]

        # Stats
        self.stats = {
            "images_processed": 0,
            "masks_identified": 0,
            "per_class_masks": defaultdict(int),
        }

        # Output directories
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.save_visualizations:
                (self.output_dir / "visualizations").mkdir(exist_ok=True)
            (self.output_dir / "masks_plas").mkdir(exist_ok=True)
            (self.output_dir / "masks_propagation").mkdir(exist_ok=True)
            (self.output_dir / "masks_propagation_plas").mkdir(exist_ok=True)
            (self.output_dir / "stats").mkdir(exist_ok=True)

        # Color mappings
        self.color_to_label = {}
        self.label_to_color = {}
        self.color_to_label[self.DEFAULT_BACKGROUND_COLOR] = self.DEFAULT_BACKGROUND_CLASS_ID
        self.label_to_color[self.DEFAULT_BACKGROUND_CLASS_ID] = self.DEFAULT_BACKGROUND_COLOR

        # Load/create mapping
        self.load_or_create_color_mapping()

        print("Unified AutoLabeler initialized with:")
        print(f"  - Images directory: {self.images_dir}")
        print(f"  - Ground truth directory: {self.ground_truth_dir}")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Device: {self.device}")
        print(f"  - Point selection strategy: {self.point_selection_strategy}")
        print(f"  - Interactive strategy: {self.is_interactive_strategy}")
        print(f"  - Number of points: {self.num_points}")
        print(f"  - Strategy-specific parameters: {self.strategy_kwargs}")
        if self.save_expanded_masks_debug:
            print("  - Expanded mask debug saving: ENABLED")

        # Lazy segmenter init
        self.segmenter = None
        self._segmenter_initialized = False

        # Timing
        self.timer = PipelineTimer()

        # Downscale config (optional multi-scale assistance)
        self.downscale_auto = downscale_auto
        self.downscale_fixed = downscale_fixed
        self.scale_config = ScaleConfig() if ScaleConfig is not None else None
        self.current_scale = 1.0
        self.coverage_map = None  # low-res coverage (uint16) if scaling active

    def _initialize_segmenter(self):
        """Initialize the segmenter when first needed."""
        if not self._segmenter_initialized:
            self.segmenter = Segmenter(
                image=None,
                sam2_checkpoint_path=self.sam2_checkpoint,
                sam2_config_path=self.sam2_config,
                device=self.device
            )
            self._segmenter_initialized = True

    def build_complete_color_mapping(self):
        """
        Build a complete color mapping by analyzing all ground truth images.
        This ensures all colors in GT images are mapped to class indices.
        """
        print("Building complete color mapping from ground truth images...")
        
        image_files = self.get_image_files()
        all_colors = set()
        
        for img_path, gt_path in tqdm(image_files, desc="Analyzing GT images", leave=False):
            gt_image = cv2.imread(str(gt_path))
            if gt_image is None:
                continue
                
            gt_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            
            # Get unique colors (sample every 4th pixel for speed)
            unique_colors = set(map(tuple, gt_rgb[::4, ::4].reshape(-1, 3)))
            all_colors.update(unique_colors)
        
        print(f"Found {len(all_colors)} unique colors in ground truth images")
        
        # Build color mapping
        self.color_to_label = {}
        self.label_to_color = {}
        
        # Always start with background
        black = (0, 0, 0)
        if black in all_colors:
            self.color_to_label[black] = 0
            self.label_to_color[0] = black
            all_colors.remove(black)
        
        # Map remaining colors to consecutive class indices
        for i, color in enumerate(sorted(all_colors), start=1):
            self.color_to_label[color] = i
            self.label_to_color[i] = color
        
        print(f"Created color mapping with {len(self.color_to_label)} entries")
        
        # Save the mapping
        self.save_color_mapping()

    def load_or_create_color_mapping(self):
        """Load existing color to label mapping or create a new one."""
        mapping_file = self.output_dir / "color_mapping.json"
        
        if mapping_file.exists():
            try:
                print(f"Loading color mapping from {mapping_file}")
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                    # Convert string keys (from JSON) back to tuple keys
                    self.color_to_label = {eval(k): v for k, v in mapping["color_to_label"].items()}
                    self.label_to_color = {k: v for k, v in mapping["label_to_color"].items()}
                print(f"Successfully loaded color mapping with {len(self.color_to_label)} entries")
            except json.JSONDecodeError as e:
                print(f"Error loading color mapping: {e}")
                print("Creating a new color mapping file.")
                self.color_to_label = {}
                self.label_to_color = {}
                # Rename the corrupted file
                backup_file = mapping_file.with_suffix('.json.bak')
                try:
                    mapping_file.rename(backup_file)
                    print(f"Backed up corrupted mapping file to {backup_file}")
                except Exception as rename_err:
                    print(f"Could not rename corrupted file: {rename_err}")
        else:
            print("No existing color mapping found. Will create during processing.")
            self.color_to_label = {}
            self.label_to_color = {}
    
    def save_color_mapping(self, verbose=False):
        """Save color to label mapping for future use."""
        mapping_file = self.output_dir / "color_mapping.json"
        
        # Convert tuple keys to strings and ensure all values are JSON serializable
        color_to_label_serializable = {str(k): v for k, v in self.color_to_label.items()}
        label_to_color_serializable = {k: [int(c) for c in v] for k, v in self.label_to_color.items()}
        
        # Create a temporary file first to avoid corrupting the existing file
        temp_file = mapping_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump({
                    "color_to_label": color_to_label_serializable,
                    "label_to_color": label_to_color_serializable
                }, f, indent=2)
                f.flush()
                import os
                os.fsync(f.fileno())  # Ensure data is written to disk
            
            # Rename the temporary file to the target file (atomic operation on most systems)
            temp_file.replace(mapping_file)
            if verbose:
                print(f"Color mapping saved to {mapping_file} ({len(self.color_to_label)} entries)")
        except Exception as e:
            print(f"Error saving color mapping: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()  # Delete the temporary file
                except Exception:
                    pass

    def get_image_files(self):
        """Get list of image files to process (with optional ground truth)."""
        if self._cached_image_files is not None:
            return self._cached_image_files

        supported_exts = {'.jpg', '.jpeg', '.png'}
        image_files = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported_exts
        )

        # Sparse GT only mode
        if self.ground_truth_dir is None:
            pairs = [(p, None) for p in image_files]
            print(f"Found {len(pairs)} images for sparseGT-only processing")
            self._cached_image_files = pairs
            return pairs

        # Build lowercase lookup for GT files once
        gt_lookup = {
            f.name.lower(): f for f in self.ground_truth_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_exts
        }

        valid = []
        for img in image_files:
            key = img.name.lower()
            gt = gt_lookup.get(key)
            if gt is None:
                base = img.stem.lower()
                for ext in ('.png', '.jpg', '.jpeg'):
                    alt = gt_lookup.get(base + ext)
                    if alt is not None:
                        gt = alt
                        break
            if gt is not None and gt.exists():
                valid.append((img, gt))
            else:
                print(f"Warning: No ground truth found for {img.name}, skipping.")

        print(f"Found {len(valid)} valid image/ground-truth pairs")
        self._cached_image_files = valid
        return valid

    def extract_labels_from_ground_truth(self, gt_image):
        """
        Extract unique labels from ground truth image.
        
        Args:
            gt_image: Ground truth image (RGB or grayscale)
            
        Returns:
            List of dictionaries with mask information
        """
        # Check if the image is grayscale or RGB
        is_grayscale = len(gt_image.shape) == 2 or (len(gt_image.shape) == 3 and gt_image.shape[2] == 1)
        
        # Process based on the image type
        if is_grayscale:
            # Convert to proper grayscale format if needed
            if len(gt_image.shape) == 3:
                gt_image = gt_image[:, :, 0]  # Use first channel
                
            # Get unique values in grayscale image
            unique_values = set()
            height, width = gt_image.shape[:2]
            
            # Skip zero (background) value
            for y in range(0, height, 4):  # Sample every 4th pixel for speed
                for x in range(0, width, 4):
                    value = int(gt_image[y, x])
                    if value > 0:  # Skip background
                        unique_values.add(value)
            
            # Create binary masks for each grayscale value/class
            labels = []  # Use a list instead of dict with numpy array keys
            for value in unique_values:
                # Create mask for this value
                value_mask = (gt_image == value)
                
                # Get or create label for this value
                value_key = ('grayscale', value)  # Use tuple to distinguish from RGB
                if value_key in self.color_to_label:
                    label = self.color_to_label[value_key]
                else:
                    # Create new label for this value - use numeric class ID
                    label = len(self.color_to_label) + 1
                    self.color_to_label[value_key] = label
                    self.label_to_color[value_key] = value  # Store as grayscale value
                
                # Store mask info
                area = np.sum(value_mask)
                if area > 100:  # Skip tiny areas
                    labels.append({
                        "mask": value_mask,
                        "label": label,
                        "color": (value, value, value),  # Use RGB format for consistency
                        "grayscale_value": value,
                        "area": area
                    })
        else:
            # Original RGB processing
            unique_colors = set(map(tuple, gt_image[::4, ::4].reshape(-1, 3)))

            # Create binary masks for each color/class
            labels = []  # Use a list instead of dict with numpy array keys
            for color in unique_colors:
                color_mask = np.all(gt_image == color, axis=2)
                # Enforce fixed background mapping
                if color == self.DEFAULT_BACKGROUND_COLOR:
                    label = self.DEFAULT_BACKGROUND_CLASS_ID
                    self.color_to_label[color] = label
                    self.label_to_color[label] = color
                else:
                    if color in self.color_to_label:
                        label = self.color_to_label[color]
                    else:
                        label = len([c for c in self.color_to_label.values() if c != self.DEFAULT_BACKGROUND_CLASS_ID]) + 1
                        # Avoid collision with reserved background id
                        if label == self.DEFAULT_BACKGROUND_CLASS_ID:
                            label += 1
                        self.color_to_label[color] = label
                        self.label_to_color[label] = color
                
                area = np.sum(color_mask)
                if area > 100:  # Skip tiny areas
                    labels.append({
                        "mask": color_mask,
                        "label": label,
                        "color": color,
                        "area": area
                    })
        
        return labels

    def find_mask_for_point(self, point, gt_masks):
        """
        Find the ground truth mask containing the given point.
        
        Args:
            point: (y, x) tuple
            gt_masks: List of dictionaries with mask information
            
        Returns:
            The mask containing the point and its label, or None if not found
        """
        y, x = point
        for mask_info in gt_masks:
            mask = mask_info["mask"]
            if y < mask.shape[0] and x < mask.shape[1] and mask[x, y]:
                return mask, mask_info["label"], mask_info["color"]
        return None, None, None

    # def merge_overlapping_masks(self, masks, mask_labels, gt_points=None, gt_labels=None):
    #     """
    #     Identical behavior to your original:
    #     • Groups pixels by the exact set of mask IDs covering them
    #     • Sorts regions by number of overlapping masks
    #     • For each region: centroid + distance-weighted GT voting
    #     • Fills non-overlaps the same
    #     • Builds label_colors exactly the same
    #     """
    #     import numpy as np
    #     from collections import defaultdict
    #     from PyQt6.QtGui import QColor

    #     if not masks:
    #         return None, None

    #     # 1) label↔ID
    #     unique_labels = sorted(set(mask_labels))
    #     label_to_id = {lbl: i+1 for i, lbl in enumerate(unique_labels)}
    #     id_to_label = {i+1: lbl for i, lbl in enumerate(unique_labels)}
    #     mask_ids = [label_to_id[lbl] for lbl in mask_labels]

    #     # 2) stack to (H, W, N) bool
    #     H, W = masks[0].shape
    #     M = np.stack(masks, axis=-1)

    #     # 3) prepare GT arrays
    #     if gt_points is not None and gt_labels is not None:
    #         gt_pts = np.array(gt_points)
    #         gt_lbls = np.array(gt_labels)
    #     else:
    #         gt_pts = gt_lbls = None

    #     # 4) build pixel‐to‐segments only for overlapping pixels
    #     cover_counts = M.sum(axis=-1)
    #     overlap_coords = np.argwhere(cover_counts > 1)

    #     # group by exact tuple of seg IDs
    #     groups = defaultdict(list)
    #     for (r, c) in overlap_coords:
    #         segs = tuple(np.where(M[r, c, :])[0].tolist())
    #         if len(segs) > 1:
    #             groups[tuple(sorted(segs))].append((r, c))

    #     # build list of (region_coords, seg_ids) sorted by len(seg_ids) desc
    #     overlap_regions = sorted(
    #         [(coords, seg_ids) for seg_ids, coords in groups.items()],
    #         key=lambda x: len(x[1]),
    #         reverse=True
    #     )

    #     # 5) init outputs
    #     final_mask      = np.zeros((H, W), dtype=np.int32)
    #     resolved_pixels = np.zeros((H, W), dtype=bool)

    #     # 6) resolve each overlapping region exactly as before
    #     for coords, seg_ids in overlap_regions:
    #         # unresolved pixels in this region
    #         unresolved = [(r, c) for (r, c) in coords if not resolved_pixels[r, c]]
    #         if not unresolved:
    #             continue

    #         pts_arr = np.array(unresolved)
    #         centroid = pts_arr.mean(axis=0)

    #         best_score = -np.inf
    #         best_sid   = None

    #         for sid in seg_ids:
    #             lbl = mask_labels[sid]
    #             if gt_pts is not None:
    #                 # pick GT pts matching this label and in unresolved & in this region
    #                 mask = (gt_lbls == lbl)
    #                 pts = gt_pts[mask]
    #                 if pts.size:
    #                     # only those inside this region and still unresolved
    #                     pts = np.array([pt for pt in pts
    #                                     if (tuple(pt) in unresolved)])
    #                 if pts.size == 0:
    #                     continue

    #                 dists   = np.linalg.norm(pts - centroid, axis=1)
    #                 weights = 1.0 / (1.0 + dists)
    #                 score   = weights.sum()
    #             else:
    #                 score = 0.0

    #             if score > best_score:
    #                 best_score = score
    #                 best_sid   = sid

    #         if best_sid is not None:
    #             rows, cols = zip(*unresolved)
    #             final_mask[rows, cols]      = label_to_id[mask_labels[best_sid]]
    #             resolved_pixels[rows, cols] = True

    #     # 7) fill non-overlapping (gaps) exactly as before
    #     for sid, mask in enumerate(masks):
    #         gap = mask & ~resolved_pixels
    #         if np.any(gap):
    #             final_mask[gap]      = mask_ids[sid]
    #             resolved_pixels[gap] = True

    #     # 8) build colors exactly as before
    #     label_colors = {}
    #     for lid in np.unique(final_mask):
    #         if lid == 0:
    #             continue
    #         orig = id_to_label[lid]
    #         clr  = getattr(self, 'label_to_color', {}).get(orig)
    #         if clr:
    #             label_colors[lid] = QColor(*clr) if isinstance(clr, tuple) \
    #                                 else QColor(clr[0], clr[1], clr[2])
    #         else:
    #             rand_rgb = tuple(np.random.randint(0, 256, 3))
    #             label_colors[lid] = QColor(*rand_rgb)

    #     return final_mask, label_colors

    def merge_overlapping_masks(self, masks, mask_labels, gt_points=None, gt_labels=None):
        """Merge overlapping SAM2 masks into a single partial segmentation map using
        a majority vote among the K nearest GT points (per overlap region).

        Simplified procedure:
        1. For each exact-overlap region (intersection of a specific covering set of masks), compute its centroid.
        2. Collect GT points whose labels are among the labels of the participating masks.
        3. Take the K nearest (self.nearest_gt_k, default 10) to the centroid and choose the majority label.
           Tie break: label with smallest mean distance among its contributing points.
        4. If no GT points available for that region, fall back to majority mask-label frequency, else largest area.
        5. Pixels covered by exactly one mask are assigned that mask's label directly.
        """
        import numpy as np

        if not masks:
            return None, None

        # 1) Label ↔ ID mapping
        unique_labels = list(dict.fromkeys(mask_labels))
        label_to_id = {lbl: i + 1 for i, lbl in enumerate(unique_labels)}
        id_to_label = {i + 1: lbl for i, lbl in enumerate(unique_labels)}
        mask_ids = [label_to_id[lbl] for lbl in mask_labels]

        # 2) Stack masks
        H, W = masks[0].shape
        stacked = np.stack(masks, axis=-1)

        # 3) GT arrays
        if gt_points is not None and gt_labels is not None:
            gt_points_arr = np.asarray(gt_points)
            gt_labels_arr = np.asarray(gt_labels)
        else:
            gt_points_arr = gt_labels_arr = None

        # 4) Group overlap regions by exact covering set
        cover_counts = stacked.sum(axis=-1)
        multi_cover_coords = np.argwhere(cover_counts > 1)
        overlap_groups = defaultdict(list)
        for (r, c) in multi_cover_coords:
            seg_indices = tuple(np.where(stacked[r, c])[0].tolist())
            if len(seg_indices) > 1:
                overlap_groups[seg_indices].append((r, c))
        overlap_regions = sorted(
            [(coords, seg_ids) for seg_ids, coords in overlap_groups.items()],
            key=lambda x: len(x[1]), reverse=True
        )

        # 5) Outputs
        final_mask = np.zeros((H, W), dtype=np.int32)
        resolved = np.zeros((H, W), dtype=bool)

        # 6) Resolve overlaps
        k_nearest = getattr(self, 'nearest_gt_k', 5)
        for coords, seg_ids in overlap_regions:
            coords_unresolved = [(r, c) for (r, c) in coords if not resolved[r, c]]
            if not coords_unresolved:
                continue
            region_pts = np.asarray(coords_unresolved)
            centroid = region_pts.mean(axis=0)
            # Decide label for this overlap region:
            # 1. If region contains any GT points (from participating labels), use Score = sum 1/(1+dist) over GT points inside region per label.
            # 2. Otherwise, use majority label among K nearest GT points (restricted to participating labels).
            best_seg = None
            region_has_gt = False
            if gt_points_arr is not None and gt_points_arr.size:
                candidate_labels = {mask_labels[s] for s in seg_ids}
                # Filter GT points to only candidate labels
                candidate_mask = np.isin(gt_labels_arr, list(candidate_labels))
                cand_pts_all = gt_points_arr[candidate_mask]
                cand_lbls_all = gt_labels_arr[candidate_mask]
                if cand_pts_all.size:
                    # Build region mask for full region (all coords, not only unresolved)
                    region_mask = np.zeros((H, W), dtype=bool)
                    for (rr, cc) in coords:
                        region_mask[rr, cc] = True
                    rows_pts = cand_pts_all[:, 0].astype(int)
                    cols_pts = cand_pts_all[:, 1].astype(int)
                    # Clamp to valid bounds (handles non-square images safely)
                    rows_pts = np.clip(rows_pts, 0, H - 1)
                    cols_pts = np.clip(cols_pts, 0, W - 1)
                    inside_region = region_mask[rows_pts, cols_pts]
                    if np.any(inside_region):
                        region_has_gt = True
                        # Scoring path
                        best_score = -np.inf
                        for seg_idx in seg_ids:
                            lbl = mask_labels[seg_idx]
                            lbl_mask = (cand_lbls_all == lbl) & inside_region
                            if not np.any(lbl_mask):
                                continue
                            pts_in = cand_pts_all[lbl_mask]
                            dists = np.linalg.norm(pts_in - centroid, axis=1)
                            if dists.size == 0:
                                continue
                            score = np.sum(1.0 / (1.0 + dists))
                            if score > best_score:
                                best_score = score
                                best_seg = seg_idx
                    # If region_has_gt but best_seg still None (pathological), fall through to other strategies
                if not region_has_gt:
                    # K-nearest majority path (region has no GT points inside it)
                    if cand_pts_all.size:
                        # Distance of all candidate label points to centroid
                        dists = np.linalg.norm(cand_pts_all - centroid, axis=1)
                        order = np.argsort(dists)
                        k_use = min(k_nearest, order.shape[0])
                        top_idx = order[:k_use]
                        top_lbls = cand_lbls_all[top_idx]
                        top_dists = dists[top_idx]
                        # Compute distance-weighted score per label: sum 1/(1+dist)
                        best_score_k = -np.inf
                        winning_label = None
                        tie_candidates = []
                        for lbl in np.unique(top_lbls):
                            lbl_mask = (top_lbls == lbl)
                            lbl_dists = top_dists[lbl_mask]
                            if lbl_dists.size == 0:
                                continue
                            score_lbl = np.sum(1.0 / (1.0 + lbl_dists))
                            if score_lbl > best_score_k + 1e-12:  # clear better
                                best_score_k = score_lbl
                                winning_label = lbl
                                tie_candidates = [(lbl, lbl_dists.mean())]
                            elif abs(score_lbl - best_score_k) <= 1e-12:  # tie
                                tie_candidates.append((lbl, lbl_dists.mean()))
                        if winning_label is not None and len(tie_candidates) > 1:
                            # Tie-break: smallest mean distance
                            tie_candidates.sort(key=lambda x: x[1])
                            winning_label = tie_candidates[0][0]
                        if winning_label is not None:
                            for seg_idx in seg_ids:
                                if mask_labels[seg_idx] == winning_label:
                                    best_seg = seg_idx
                                    break

            if best_seg is None:
                label_counts = {}
                for seg_idx in seg_ids:
                    lbl = mask_labels[seg_idx]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                if label_counts:
                    majority_label, majority_count = max(label_counts.items(), key=lambda x: (x[1], x[0]))
                    if list(label_counts.values()).count(majority_count) == 1 and majority_count > 1:
                        for seg_idx in seg_ids:
                            if mask_labels[seg_idx] == majority_label:
                                best_seg = seg_idx
                                break
                if best_seg is None:
                    max_area = -1
                    for seg_idx in seg_ids:
                        area = sum(stacked[r, c, seg_idx] for (r, c) in coords_unresolved)
                        if area > max_area:
                            max_area = area
                            best_seg = seg_idx

            chosen_label_id = label_to_id[mask_labels[best_seg]]
            rows, cols = zip(*coords_unresolved)
            final_mask[rows, cols] = chosen_label_id
            resolved[rows, cols] = True

        # 7) Single-mask coverage
        for seg_idx, mask in enumerate(masks):
            single_area = mask & ~resolved
            if not np.any(single_area):
                continue
            final_mask[single_area] = mask_ids[seg_idx]
            resolved[single_area] = True

        # 8) QColor mapping
        label_colors = {}
        for lid in np.unique(final_mask):
            if lid == 0:
                continue
            orig_label = id_to_label[lid]
            clr_map = getattr(self, 'label_to_color', {})
            clr = clr_map.get(orig_label) if isinstance(clr_map, dict) else None
            if clr is not None:
                label_colors[lid] = QColor(*clr) if isinstance(clr, tuple) else QColor(clr[0], clr[1], clr[2])
            else:
                rand = tuple(np.random.randint(0, 256, 3))
                label_colors[lid] = QColor(*rand)

        return final_mask, label_colors

    def extract_sam2_features(self, mask=None):
        """
        Extract SAM2 features from a masked region.
        
        Args:
            mask: Binary mask to focus on a specific region
            
        Returns:
            Feature vector from SAM2
        """
        if self.features_sam2 is None:
            return None
            
        mask_tensor = torch.from_numpy(mask).to(self.features_sam2.device)        # [H_orig, W_orig]
        # Resize mask to feature spatial size using nearest neighbor
        mask_resized = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=self.features_sam2.shape[-2:], mode='nearest'
        ).squeeze(0).squeeze(0)  # Now shape [H_feat, W_feat]
        # Apply mask: broadcast to all channels
        masked_feats = self.features_sam2 * mask_resized
        # Flatten spatial dims: each channel's values for the masked region
        masked_flat = masked_feats.view(self.features_sam2.size(0), -1)            # [C, H_feat*W_feat]
        # Also flatten mask to count region size
        mask_flat = mask_resized.view(-1)                               # [H_feat*W_feat]
        # Select only the entries within the mask (mask_flat == 1)
        if mask_flat.sum() > 0:
            region_feats = masked_flat[:, mask_flat.bool()]             # [C, N_pixels]
            region_descriptor = region_feats.mean(dim=1)                # [C] vector
        else:
            region_descriptor = torch.zeros(self.features_sam2.size(0), device=self.features_sam2.device)

        return region_descriptor

    def process_image_interactive(self, image_path, gt_path, color_dict=None):
        """
        Process image using interactive/iterative point selection strategies.
        """
        self.timer.reset()
        self.timer.start_total()

        # Initialize segmenter if not already done
        self._initialize_segmenter()

        # Load image
        with self.timer.time_operation('io_operations'):
            image = cv2.imread(str(image_path))

        # Decide auxiliary scale (never affects SAM2 embedding resolution)
        self.current_scale = 1.0
        self.coverage_map = None
        if (self.downscale_fixed is not None or self.downscale_auto) and choose_scale is not None:
            if self.downscale_fixed is not None:
                self.current_scale = float(max(0.1, min(1.0, self.downscale_fixed)))
            elif self.downscale_auto and self.scale_config is not None:
                try:
                    self.current_scale = float(choose_scale(image.shape, self.scale_config))
                except Exception:
                    self.current_scale = 1.0
            if self.current_scale < 0.999 and init_coverage_map is not None:
                low_h = max(1, int(round(image.shape[0] * self.current_scale)))
                low_w = max(1, int(round(image.shape[1] * self.current_scale)))
                self.coverage_map = init_coverage_map((low_h, low_w))

        with self.timer.time_operation('setup'):
            generated_masks, self.features_sam2 = self.segmenter.set_image(image)

            generated_masks_overlay = np.zeros(image.shape[:2], dtype=np.uint8)
            # Create overlay for generated masks
            for mask in generated_masks:
                generated_masks_overlay[mask['segmentation'] > 0] = 1

        # Handle ground truth loading (optional for sparseGT-only mode)
        gt_masks = []
        if gt_path is not None:
            with self.timer.time_operation('io_operations'):
                # Load ground truth
                gt_image = cv2.imread(str(gt_path))
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
                
                # Extract ground truth masks/labels
                gt_masks = self.extract_labels_from_ground_truth(gt_image)
        else:
            # sparseGT-only mode: no ground truth available
            # Suppressed per-image print to keep tqdm progress bar clean
            pass

        # Track expanded masks and their labels
        expanded_masks = []
        
        # Setup for interactive strategy
        with self.timer.time_operation('setup'):
            if hasattr(self.strategy, 'set_gt_masks'):
                try:
                    self.strategy.set_gt_masks(gt_masks)
                except Exception:
                    pass
            self.strategy.setup_simple(image, generated_masks)
        
        points_to_process = []
        labels_to_process = []
        rgb_to_int_label = {}
        int_labels_to_rgb = {}
        next_label = 1

        last_mask = None

        # Interactive point selection loop
        for i in range(self.num_points):
            with self.timer.time_operation('point_selection'):
                pt = self.strategy.get_next_point(last_mask)
                
                if pt is None:
                    continue
                    
                x, y = pt[1], pt[0]
                current_point = (x, y)
                
                points_to_process.append(current_point)
                rgb_label = tuple(gt_image[pt[0], pt[1]])
                if rgb_label not in rgb_to_int_label:
                    rgb_to_int_label[rgb_label] = next_label
                    int_labels_to_rgb[next_label] = rgb_label
                    next_label += 1
                labels_to_process.append(rgb_to_int_label[rgb_label])
                
                # Find which ground truth mask contains this point
                gt_mask, gt_label, gt_color = self.find_mask_for_point(current_point, gt_masks)
                
                if gt_mask is None:
                    continue
            
            with self.timer.time_operation('sam2_propagation'):
                # Expand the mask using SAM2
                point_pair = np.array([current_point])
                label_pair = np.array([1])  # Positive point
                mask = self.segmenter.propagate_points(point_pair, label_pair, update_expanded_mask=True)
                last_mask = mask  # Update last_mask for the next iteration
                
                if mask is None:
                    continue
                
                # Add to expanded masks list
                # Store (mask, label, color, seed_point)
                expanded_masks.append((mask, gt_label, gt_color, current_point))
                self.stats["masks_identified"] += 1
                self.stats["per_class_masks"][gt_label] += 1

                # Update coverage (best-effort)
                if self.coverage_map is not None and update_coverage_map is not None:
                    try:
                        update_coverage_map(self.coverage_map, mask, self.current_scale, bg_value=0)
                    except Exception:
                        pass

        return self._finalize_processing_with_timing(
            image_path, image, gt_image if gt_path is not None else None, points_to_process, labels_to_process, 
            int_labels_to_rgb, expanded_masks, color_dict
        )

    def process_image_batch(self, image_path, gt_path, color_dict=None):
        """
        Process image using batch/traditional point selection strategies.
        """
        self.timer.reset()
        self.timer.start_total()

        # Initialize segmenter if not already done
        self._initialize_segmenter()

        # Load image
        with self.timer.time_operation('io_operations'):
            image = cv2.imread(str(image_path))
        # Decide scale for batch path
        self.current_scale = 1.0
        self.coverage_map = None
        if (self.downscale_fixed is not None or self.downscale_auto) and choose_scale is not None:
            if self.downscale_fixed is not None:
                self.current_scale = float(max(0.1, min(1.0, self.downscale_fixed)))
            elif self.downscale_auto and self.scale_config is not None:
                try:
                    self.current_scale = float(choose_scale(image.shape, self.scale_config))
                except Exception:
                    self.current_scale = 1.0
            if self.current_scale < 0.999 and init_coverage_map is not None:
                low_h = max(1, int(round(image.shape[0] * self.current_scale)))
                low_w = max(1, int(round(image.shape[1] * self.current_scale)))
                self.coverage_map = init_coverage_map((low_h, low_w))
            
        with self.timer.time_operation('setup'):
            self.segmenter.just_set_image(image)

        # Handle ground truth loading (optional for sparseGT-only mode)
        gt_masks = []
        if gt_path is not None:
            with self.timer.time_operation('io_operations'):
                # Load ground truth
                gt_image = cv2.imread(str(gt_path))
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
                
                # Extract ground truth masks/labels
                gt_masks = self.extract_labels_from_ground_truth(gt_image)
        else:
            # sparseGT-only mode: no ground truth available
            # Suppressed per-image print to keep tqdm progress bar clean
            pass
        
        # Track expanded masks and their labels
        expanded_masks = []
        
        with self.timer.time_operation('point_selection'):
            # Get points and classes to process using the unified strategy
            points_and_classes = self.strategy.select_points(
                self.segmenter,
                image, 
                gt_masks,  # Will be empty list for sparseGT-only mode
                expanded_masks=expanded_masks,
                image_name=Path(image_path).stem
            )

            # Handle the unified return value (points, classes)
            if isinstance(points_and_classes, tuple) and len(points_and_classes) == 2:
                points_to_process, point_classes = points_and_classes
            else:
                # Fallback for strategies that only return points
                points_to_process = points_and_classes
                point_classes = None
            
            # Process points and create labels
            labels_to_process = []
            rgb_to_int_label = {}
            int_labels_to_rgb = {}
            next_label = 1 
            
            for i, point in enumerate(points_to_process):
                if point is None:
                    continue

                if gt_path is not None:
                    # RGB ground truth mode: extract RGB label from ground truth
                    rgb_label = tuple(gt_image[point[0], point[1]])  # Convert to tuple for hashability

                    # Check if this RGB value is already mapped
                    if rgb_label not in rgb_to_int_label:
                        # Assign a new unique integer label
                        rgb_to_int_label[rgb_label] = next_label
                        int_labels_to_rgb[next_label] = rgb_label
                        next_label += 1

                    # Add the corresponding integer label to the list
                    labels_to_process.append(rgb_to_int_label[rgb_label])
                else:
                    # sparseGT-only mode: use class information from points
                    if point_classes is None or i >= len(point_classes):
                        raise ValueError(f"sparseGT-only mode requires class information for point {i}")
                    point_class = point_classes[i]
                    labels_to_process.append(point_class)
            
            # Process each selected point; for 'list' strategy we keep all provided points
            if self.point_selection_strategy != 'list':
                points_to_process = points_to_process[:self.num_points]
                labels_to_process = labels_to_process[:self.num_points]
            else:
                pass

            #flip points coordinates
            points_to_process = np.array(points_to_process)[:, [1, 0]]

        iteration = 0
        for idx, current_point in enumerate(points_to_process):
            iteration += 1
            if current_point is None:
                continue

            point_label = labels_to_process[idx] if idx < len(labels_to_process) else None
            gt_label = None
            gt_color = None

            if gt_masks:
                with self.timer.time_operation('point_selection'):
                    gt_mask, gt_label, gt_color = self.find_mask_for_point(current_point, gt_masks)
            # Always propagate regardless of gt_mask presence
            with self.timer.time_operation('sam2_propagation'):
                point_pair = np.array([current_point])
                label_pair = np.array([1])
                mask = self.segmenter.propagate_points(point_pair, label_pair, update_expanded_mask=True)
                if mask is None:
                    continue

                eff_label = gt_label if gt_label is not None else point_label
                if eff_label is None:
                    continue

                # Resolve color
                if gt_color is None:
                    # Try existing mapping
                    if eff_label in int_labels_to_rgb:
                        gt_color = tuple(int_labels_to_rgb[eff_label])
                    # Try color_dict inversion (sparseGT-only)
                    elif color_dict is not None:
                        inv = getattr(self, '_cached_inv_color_dict', None)
                        if inv is None:
                            inv = {}
                            for rgb, cls in color_dict.items():
                                inv.setdefault(cls, rgb)
                            self._cached_inv_color_dict = inv
                        if eff_label in inv:
                            gt_color = inv[eff_label]
                            int_labels_to_rgb.setdefault(eff_label, list(gt_color))
                    # Fallback stable random
                    if gt_color is None:
                        rng = np.random.default_rng(eff_label)
                        gt_color = tuple(int(x) for x in rng.integers(0,256,3))
                        int_labels_to_rgb.setdefault(eff_label, list(gt_color))

                expanded_masks.append((mask, eff_label, gt_color, tuple(current_point)))
                self.stats["masks_identified"] += 1
                self.stats["per_class_masks"][eff_label] += 1

        return self._finalize_processing_with_timing(
            image_path, image, gt_image if gt_path is not None else None, points_to_process, labels_to_process, 
            int_labels_to_rgb, expanded_masks, color_dict
        )

    def _finalize_processing_with_timing(self, image_path, image, gt_image, points_to_process, 
                                       labels_to_process, int_labels_to_rgb, expanded_masks, color_dict=None, gt_masks=None):
        """Finalize processing: unify masks, apply background, return result dict."""
        # 1. Determine classes
        if self.num_classes is None:
            self._determine_num_classes(color_dict, labels_to_process, allow_infer=True)

        # 2. Sync color maps
        for lbl, rgb in int_labels_to_rgb.items():
            t = tuple(rgb)
            self.color_to_label.setdefault(t, lbl)
            self.label_to_color.setdefault(lbl, rgb)
        background_color = list(self.DEFAULT_BACKGROUND_COLOR)

        # 3. Visualizations (optional)
        if self.save_visualizations:
            vis_dir = self.output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            for i, (x, y) in enumerate(points_to_process):
                if i < len(labels_to_process):
                    lbl = labels_to_process[i]
                    c = np.array(int_labels_to_rgb.get(lbl, background_color)) / 255.0
                    plt.scatter(x, y, color=c, s=150, edgecolors='yellow', linewidths=2.0)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(vis_dir / f"{Path(image_path).stem}.png", bbox_inches='tight')
            plt.close()

        # 4. PLAS mask
        plas_mask = self.PLAS_segmenter.expand_labels(
            image, points_to_process, labels_to_process,
            num_classes=self.num_classes)
        plas_mask_rgb = np.full((image.shape[0], image.shape[1], 3), background_color, dtype=np.uint8)
        for lbl, rgb in int_labels_to_rgb.items():
            plas_mask_rgb[plas_mask == lbl] = rgb

        # 5. Merge SAM propagation masks
        if expanded_masks:
            # Use points_to_process (likely in (x,y)) and labels_to_process directly as GT evidence
            # Vectorized normalization to (row, col) = (y, x)
            pts_arr = np.asarray(points_to_process)
            if pts_arr.size == 0:
                gt_pts_arr = np.empty((0, 2), dtype=np.int32)
            else:
                if pts_arr.ndim != 2 or pts_arr.shape[1] != 2:
                    pts_arr = np.array([tuple(p) for p in points_to_process], dtype=np.int32)
                else:
                    pts_arr = pts_arr.astype(np.int32, copy=False)
                # swap columns: (x,y) -> (y,x)
                gt_pts_arr = pts_arr[:, [1, 0]]
            gt_labs_arr = np.asarray(labels_to_process, dtype=np.int32)
            # Optionally, if ground truth image is present, extend with dense GT mask points
            if gt_image is not None:
                reuse_masks = gt_masks if gt_masks is not None else (self.extract_labels_from_ground_truth(gt_image) or [])
                for mi in reuse_masks:
                    pts = np.argwhere(mi['mask'])  # (row,col)
                    if pts.size == 0:
                        continue
                    if pts.shape[0] <= self.min_gt_points_full_mask:
                        sampled = pts
                    else:
                        sample_size = min(self.max_gt_points_per_mask, pts.shape[0])
                        # Deterministic sampling of dense GT points (seeded by GLOBAL_SEED + label)
                        try:
                            _seed_base = int(os.getenv('GLOBAL_SEED', '42'))
                        except Exception:
                            _seed_base = 42
                        _rng_dense = np.random.default_rng((_seed_base + int(mi['label'])) & 0xFFFFFFFF)
                        idxs = _rng_dense.choice(pts.shape[0], sample_size, replace=False)
                        sampled = pts[idxs]
                    # pts from argwhere are already (row, col); collect as array and labels vector
                    sampled = sampled.astype(np.int32, copy=False)
                    if gt_pts_arr.size == 0:
                        gt_pts_arr = sampled
                    else:
                        gt_pts_arr = np.vstack((gt_pts_arr, sampled))
                    if gt_labs_arr.size == 0:
                        gt_labs_arr = np.full(sampled.shape[0], int(mi['label']), dtype=np.int32)
                    else:
                        gt_labs_arr = np.concatenate((gt_labs_arr, np.full(sampled.shape[0], int(mi['label']), dtype=np.int32)))
            # Ensure points are within image bounds (avoid OOB on non-square images)
            H_img, W_img = image.shape[:2]
            if gt_pts_arr.size:
                gt_pts_arr[:, 0] = np.clip(gt_pts_arr[:, 0], 0, H_img - 1)
                gt_pts_arr[:, 1] = np.clip(gt_pts_arr[:, 1], 0, W_img - 1)

            propagation_mask, color_map = self.merge_overlapping_masks(
                [m for m, _, _, _ in expanded_masks],
                [l for _, l, _, _ in expanded_masks],
                gt_pts_arr, gt_labs_arr
            )
            propagation_output_mask = np.full((image.shape[0], image.shape[1], 3), background_color, dtype=np.uint8)
            for lbl in np.unique(propagation_mask):
                if lbl == 0:
                    continue
                c = color_map.get(lbl)
                if c is None:
                    continue
                propagation_output_mask[propagation_mask == lbl] = [c.red(), c.green(), c.blue()]
        else:
            propagation_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
            propagation_output_mask = np.full((image.shape[0], image.shape[1], 3), background_color, dtype=np.uint8)
            for m, _, c, _ in expanded_masks:
                propagation_output_mask[m > 0] = list(c)

        # 6. Combined propagation+PLAS
        propagation_plas_mask_rgb = propagation_output_mask.copy()
        unlabeled = (propagation_mask == 0)
        propagation_plas_mask_rgb[unlabeled] = plas_mask_rgb[unlabeled]

        # Debug: save each expanded mask separately to inspect overlaps
        if self.output_dir and self.save_expanded_masks_debug and expanded_masks:
            debug_dir = self.output_dir / "expanded_masks_debug" / Path(image_path).stem
            debug_dir.mkdir(parents=True, exist_ok=True)
            H, W = image.shape[:2]
            # Save individual masks
            for idx, (m, lbl, col, seed_pt) in enumerate(expanded_masks):
                rgb = np.full((H, W, 3), background_color, dtype=np.uint8)
                color_tuple = tuple(int(x) for x in col) if isinstance(col, (list, tuple)) else background_color
                rgb[m > 0] = color_tuple
                # Draw seed point (x,y) as a small cross (white with black outline)
                if seed_pt is not None:
                    sx, sy = int(seed_pt[0]), int(seed_pt[1])  # stored as (x,y)
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            px, py = sx + dx, sy + dy
                            if 0 <= px < W and 0 <= py < H:
                                rgb[py, px] = (255, 255, 255)
                    # center pixel brighter
                    if 0 <= sx < W and 0 <= sy < H:
                        rgb[sy, sx] = (255, 0, 0)
                cv2.imwrite(str(debug_dir / f"mask_{idx:03d}_label{lbl}_pt.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            try:
                # Overlap count heatmap (how many masks cover each pixel)
                stack = np.stack([m.astype(np.uint8) for m, _, _, _ in expanded_masks], axis=0)
                cover = stack.sum(axis=0)
                cv2.imwrite(str(debug_dir / "overlap_count.png"), cover.astype(np.uint16))
                # Also a color visualization for overlaps >1
                overlap_vis = np.zeros((H, W, 3), dtype=np.uint8)
                # Simple scheme: 0 background_color, 1 keep background_color, >=2 -> red intensity
                overlap_vis[:] = background_color
                more = cover > 1
                if np.any(more):
                    # Scale intensity by (count-1)
                    max_extra = max(1, int(cover.max() - 1))
                    norm = ((cover - 1) / max_extra * 255).clip(0, 255).astype(np.uint8)
                    overlap_vis[more] = np.stack([norm[more], np.zeros_like(norm[more]), np.zeros_like(norm[more])], axis=1)
                cv2.imwrite(str(debug_dir / "overlap_visual.png"), cv2.cvtColor(overlap_vis, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"[DEBUG] Failed to write overlap debug masks: {e}")

        times = self.timer.get_pipeline_times()
        result = {
            'image_path': Path(image_path),
            'expanded_masks': expanded_masks,
            'plas_mask': plas_mask_rgb,
            'plas_mask_indices': plas_mask,
            'propagation_plas_mask': propagation_plas_mask_rgb,
            'propagation_plas_mask_indices': propagation_plas_mask_rgb[..., 0] * 0,
            'propagation_mask': propagation_output_mask,
            'propagation_mask_indices': propagation_mask,
            'plas_time': times['plas_pipeline'],
            'propagation_time': times['sam2_propagation_pipeline'],
            'propagation_plas_time': times['combined_pipeline'],
            'timing_breakdown': times['breakdown'],
            'total_time': times['total_measured'],
            'iterations': len(expanded_masks)
        }
        if 'propagation_mask_indices' in result:
            comb = result['propagation_mask_indices'].copy()
            comb[(comb == 0)] = plas_mask[(comb == 0)]
            result['propagation_plas_mask_indices'] = comb

        # 7. Remap background (only where true GT background color present)
        if self.ground_truth_dir is not None and gt_image is not None:
            # Use already loaded gt_image instead of re-reading from disk
            gt_rgb = gt_image
            bg_mask = np.all(gt_rgb == np.array(self.DEFAULT_BACKGROUND_COLOR, dtype=np.uint8), axis=2)
            for key in ['plas_mask_indices', 'propagation_mask_indices', 'propagation_plas_mask_indices']:
                arr = result.get(key)
                if isinstance(arr, np.ndarray) and arr.shape == bg_mask.shape:
                    arr[(arr == 0) & bg_mask] = self.DEFAULT_BACKGROUND_CLASS_ID
                    result[key] = arr

        # 8. Persist mapping
        self.color_to_label[self.DEFAULT_BACKGROUND_COLOR] = self.DEFAULT_BACKGROUND_CLASS_ID
        self.label_to_color[self.DEFAULT_BACKGROUND_CLASS_ID] = self.DEFAULT_BACKGROUND_COLOR
        return result

    def process_image(self, image_path, gt_path, color_dict=None):
        """
        Process a single image using the appropriate method based on strategy type.
        """
        if self.is_interactive_strategy:
            return self.process_image_interactive(image_path, gt_path, color_dict)
        else:
            return self.process_image_batch(image_path, gt_path, color_dict)

    def image_to_grayscale(self, image_bgr, color_dict=None):
        """Convert RGB mask image to class index grayscale using current color mapping.

        Unmatched pixels are assigned the default background class id (34).
        """
        color_to_label = color_dict if color_dict is not None else self.color_to_label
        if image_bgr is None:
            return None
        rgb_img = image_bgr[..., ::-1]
        h, w, _ = rgb_img.shape
        if not color_to_label:
            return np.full((h, w), self.DEFAULT_BACKGROUND_CLASS_ID, dtype=np.uint8)

        rgb_colors = np.array(list(color_to_label.keys()), dtype=np.int32)
        class_ids = np.array(list(color_to_label.values()), dtype=np.int32)
        flat = rgb_img.reshape(-1, 3)
        matches = (flat[:, None, :] == rgb_colors[None, :, :]).all(axis=2)
        any_match = matches.any(axis=1)
        idx = matches.argmax(axis=1)
        mapped = class_ids[idx]
        mapped[~any_match] = self.DEFAULT_BACKGROUND_CLASS_ID
        return mapped.reshape(h, w).astype(np.uint8)

    def save_grayscale_results(self, result, image_index, color_dict=None):
        """Save grayscale mask images derived from colored masks using external color_dict if provided.

        Mapping rules:
        - If color_dict provided: treat it as authoritative mapping { (R,G,B) : class_id }.
        - If not: fall back to self.color_to_label (dynamic mapping).
        - Background color always maps to DEFAULT_BACKGROUND_CLASS_ID (34).
        - Any RGB not found in mapping -> background class id.
        - color_dict is never mutated.
        """
        image_name = result['image_path'].stem
        out_dirs = {
            'plas': (self.output_dir / 'masks_plas', self.output_dir / 'masks_plas_gray', 'plas_mask'),
            'propagation_plas': (self.output_dir / 'masks_propagation_plas', self.output_dir / 'masks_propagation_plas_gray', 'propagation_plas_mask'),
            'propagation': (self.output_dir / 'masks_propagation', self.output_dir / 'masks_propagation_gray', 'propagation_mask')
        }
        for _, (_, gray_dir, _) in out_dirs.items():
            gray_dir.mkdir(exist_ok=True)
        # Prepare mapping (copy to avoid mutating external dict)
        if color_dict is not None:
            mapping = {}
            for k, v in color_dict.items():
                color = None
                if isinstance(k, str):
                    ks = k.strip()
                    ks = ks.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
                    parts = [p.strip() for p in ks.split(',') if p.strip()]
                    if len(parts) == 3:
                        try:
                            color = tuple(int(p) for p in parts)
                        except ValueError:
                            continue
                elif isinstance(k, (list, tuple)) and len(k) == 3:
                    try:
                        color = tuple(int(x) for x in k)
                    except ValueError:
                        continue
                if color is not None:
                    mapping[color] = int(v)
        else:
            mapping = dict(self.color_to_label)
        # Enforce background mapping
        mapping[self.DEFAULT_BACKGROUND_COLOR] = self.DEFAULT_BACKGROUND_CLASS_ID
        # Build arrays
        rgb_colors = np.array(list(mapping.keys()), dtype=np.int32)
        class_ids = np.array(list(mapping.values()), dtype=np.int32)

        for key, (color_dir, gray_dir, color_field) in out_dirs.items():
            if color_field not in result:
                continue
            color_img_rgb = result[color_field]
            if color_img_rgb is None:
                continue
            h, w, _ = color_img_rgb.shape
            flat = color_img_rgb.reshape(-1, 3).astype(np.int32)
            if rgb_colors.size == 0:
                gray = np.full((h, w), self.DEFAULT_BACKGROUND_CLASS_ID, dtype=np.uint8)
            else:
                matches = (flat[:, None, :] == rgb_colors[None, :, :]).all(axis=2)
                any_match = matches.any(axis=1)
                idx = matches.argmax(axis=1)
                mapped = class_ids[idx]
                mapped[~any_match] = self.DEFAULT_BACKGROUND_CLASS_ID
                gray = mapped.reshape(h, w).astype(np.uint8)
            cv2.imwrite(str(gray_dir / f"{image_name}.png"), gray)
    
    def save_results(self, result, image_index, color_dict=None):
        """Save colored and grayscale outputs for a processed image."""
        image_name = result['image_path'].stem
        # Ensure output dirs exist
        for sub in ["masks_plas", "masks_propagation_plas", "masks_propagation"]:
            (self.output_dir / sub).mkdir(exist_ok=True)
        # Colored masks
        cv2.imwrite(str(self.output_dir / "masks_plas" / f"{image_name}.png"), cv2.cvtColor(result['plas_mask'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.output_dir / "masks_propagation_plas" / f"{image_name}.png"), cv2.cvtColor(result['propagation_plas_mask'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.output_dir / "masks_propagation" / f"{image_name}.png"), cv2.cvtColor(result['propagation_mask'], cv2.COLOR_RGB2BGR))
        # Grayscale variants
        self.save_grayscale_results(result, image_index, color_dict)
    
    def save_stats(self):
        """Save statistics about the processed images with detailed timing breakdown."""
        # Calculate statistics for each timing type
        timing_stats = {}
        if self.stats.get("timing_breakdowns"):
            # Aggregate timing breakdown across all images
            operation_times = defaultdict(list)
            for breakdown in self.stats["timing_breakdowns"]:
                for operation, time_val in breakdown.items():
                    operation_times[operation].append(time_val)
            
            timing_stats = {
                operation: {
                    "mean": float(np.mean(times)),
                    "std": float(np.std(times)),
                    "total": float(np.sum(times))
                }
                for operation, times in operation_times.items()
            }

        # Calculate pipeline timing statistics
        pipeline_stats = {}
        for pipeline_type in ["propagation_times", "plas_times", "propagation_plas_times", "total_times"]:
            if pipeline_type in self.stats and self.stats[pipeline_type]:
                times = self.stats[pipeline_type]
                pipeline_stats[pipeline_type] = {
                    "mean": float(np.mean(times)),
                    "std": float(np.std(times)),
                    "min": float(np.min(times)),
                    "max": float(np.max(times))
                }
        
        # Save comprehensive stats as JSON
        with open(self.output_dir / "stats" / "processing_stats.json", 'w') as f:
            # Convert defaultdict to regular dict for serialization
            serializable_stats = {
                "images_processed": self.stats["images_processed"],
                "masks_identified": self.stats["masks_identified"],
                "pipeline_timing": pipeline_stats,
                "operation_timing": timing_stats,
                "masks_per_internal_class": dict(self.stats["per_class_masks"])
            }
            json.dump(serializable_stats, f, indent=2)
            
        # Print summary with improved timing information
        print("\n" + "="*100)
        print("PROCESSING SUMMARY")
        print("="*100)
        print(f"Images processed: {self.stats['images_processed']}")
        print(f"Total masks identified: {self.stats['masks_identified']}")
        
        # Print timing summary if available
        if pipeline_stats:
            print("\nTiming Summary (per image):")
            for pipeline_type, stats in pipeline_stats.items():
                pipeline_name = pipeline_type.replace("_times", "").replace("_", " ").title()
                print(f"  {pipeline_name}: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
        
        if timing_stats:
            print("\nOperation Breakdown (total across all images):")
            for operation, stats in timing_stats.items():
                operation_name = operation.replace("_", " ").title()
                print(f"  {operation_name}: {stats['total']:.3f}s (avg: {stats['mean']:.3f}s)")
        
        print("="*100)
    
    def _determine_num_classes(self, color_dict=None, class_labels=None, allow_infer=True):
        """Determine number of classes with optional inference.
        Priority: color_dict > explicit (constructor) > inferred from labels (if allow_infer).
        """
        determined = None
        sources = []
        if color_dict:
            vals = list(color_dict.values())
            determined = len(set(vals))
            sources.append(f"color_dict({determined})")
        elif self.num_classes is not None:
            determined = self.num_classes
            sources.append(f"explicit({self.num_classes})")
        if class_labels and len(class_labels) > 0:
            max_label = max(class_labels)
            min_label = min(class_labels)
            required = max_label + 1 if min_label == 0 else max_label + 1
            if determined is None and allow_infer:
                determined = required
                sources.append(f"inferred({required})")
            elif determined is None and not allow_infer:
                raise ValueError("num_classes not provided and inference disabled")
            elif determined < required:
                raise ValueError(f"num_classes {determined} < needed {required} (max label {max_label})")
        if determined is None:
            if allow_infer:
                # fallback minimal 1
                determined = 1
                sources.append("default(1)")
            else:
                raise ValueError("Unable to determine num_classes")
        self.num_classes = determined
        return determined
    
    def _validate_input_configuration(self, color_dict):
        """
        Validate that we have the correct combination of inputs before processing any images.
        """
        # Strategy capability detection (fast path)
        point_classes_available = False
        if hasattr(self.strategy, 'has_class_info'):
            try:
                point_classes_available = bool(self.strategy.has_class_info())
            except Exception:
                point_classes_available = False
        else:
            # Fallback to sampling first image (legacy behavior)
            point_classes_available = False
            if hasattr(self.strategy, 'select_points'):
                image_files = self.get_image_files()
                probe_names = []
                if image_files:
                    probe_names.append(image_files[0][0].stem)
                probe_names.append("dummy")
                for name in probe_names:
                    try:
                        points_and_classes = self.strategy.select_points(
                            segmenter=None,
                            image=None,
                            gt_masks=[],
                            image_name=name
                        )
                        if isinstance(points_and_classes, tuple) and len(points_and_classes) == 2 and points_and_classes[1] is not None:
                            point_classes_available = True
                            break
                    except Exception:
                        continue
        
        # Check if we're in RGB ground truth mode or sparseGT-only mode
        is_sparse_gt_only = self.ground_truth_dir is None
        
        if is_sparse_gt_only:  # sparseGT-only mode
            if color_dict is None:
                raise ValueError("sparseGT-only mode requires a color_dict to be provided")
            if not point_classes_available:
                raise ValueError(
                    "sparseGT-only mode (no ground truth) requires points file with class information.\n"
                    "Detected strategy: {}. It did not report a class/label column.\n"
                    "Ensure CSV has a column named one of: class, label, class_id, category.\n"
                    "Points must have format rows with (row/y, col/x, class).".format(self.strategy.__class__.__name__)
                )
            print("✓ Configuration validated: Running in sparseGT-only mode with class information")
        else:  # RGB ground truth mode
            if point_classes_available:
                print("⚠ Warning: Points file contains class information, but it will be ignored in RGB ground truth mode")
            print("✓ Configuration validated: Running in RGB ground truth mode")
            
    def process_all_images(self, color_dict=None):
        """Process all valid images in the dataset."""
        # Validate config first
        self._validate_input_configuration(color_dict)
        # Pre-determine num_classes only if explicit
        if self.num_classes is not None:
            _ = self._determine_num_classes(color_dict, class_labels=None)
        elif color_dict is not None:
            self.num_classes = self._determine_num_classes(color_dict, class_labels=None)
        # Gather images
        image_files = self.get_image_files()
        print(f"Processing {len(image_files)} images with num_classes={self.num_classes if self.num_classes is not None else 'auto'}...\n")
        # Initialize stats containers
        self.stats.update({
            "propagation_times": [],
            "plas_times": [],
            "propagation_plas_times": [],
            "total_times": [],
            "timing_breakdowns": [],
            "images_processed": 0
        })

        # Init PLAS
        self.PLAS_segmenter = SuperpixelLabelExpander(self.device, seed=self.seed)
        # Loop
        for i, (image_path, gt_path) in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                result = self.process_image(image_path, gt_path, color_dict)
                self.save_results(result, i, color_dict)
                self.save_color_mapping()
                self.stats["images_processed"] += 1
                self.stats["propagation_times"].append(result["propagation_time"])
                self.stats["plas_times"].append(result["plas_time"])
                self.stats["propagation_plas_times"].append(result["propagation_plas_time"])
                self.stats["total_times"].append(result["total_time"])
                self.stats["timing_breakdowns"].append(result["timing_breakdown"])
                self.stats.setdefault("scales", []).append(getattr(self, 'current_scale', 1.0))
            except Exception as e:
                import traceback
                print(f"\nError processing {image_path.name}: {e}")
                traceback.print_exc()
        # Aggregate timing
        if self.stats["total_times"]:
            mean_total_time = np.mean(self.stats["total_times"])
            std_total_time = np.std(self.stats["total_times"])
            print(f"\nMean total time per image: {mean_total_time:.2f}s (±{std_total_time:.2f})")
        # Persist
        self.save_color_mapping()
        self.save_stats()
        # Evaluation guard
        if self.ground_truth_dir is not None and color_dict is not None:
            if self.num_classes is None:
                observed_labels = list(self.stats["per_class_masks"].keys())
                self._determine_num_classes(color_dict, class_labels=observed_labels)
            print("\nEvaluating segmentation results...")
            self.evaluate_segmentation_results(self.num_classes, color_dict)
        else:
            print("\nSkipping evaluation (no ground truth or no color_dict).")

    def run(self, color_dict=None):
        self.process_all_images(color_dict)
    def evaluate_segmentation_results(self, num_classes, color_dict):
        """Legacy evaluation with per-class mPA & mIoU (torchmetrics) + run.py JSON.

        Mirrors previous AutoLabeler evaluation while also emitting
        stats/segmentation_metrics.json for aggregation.
        """
        print("\n" + "="*100)
        print("EVALUATING SEGMENTATION RESULTS")
        print("="*100)
        if self.ground_truth_dir is None:
            print("Skipping evaluation - no ground truth directory provided")
            print("="*100)
            return None
        if color_dict is None:
            print("Skipping evaluation - no external color dictionary provided")
            print("Provide --color-dict for consistent class mapping")
            print("="*100)
            return None

        grayscale_dir = self.output_dir / f"masks_{self.eval_mask_type}_gray"
        if not grayscale_dir.exists():
            print("No grayscale masks found for evaluation.")
            print("="*100)
            return None

        pred_files = list(grayscale_dir.glob("*.png"))
        if not pred_files:
            print("No prediction files found.")
            print("="*100)
            return None

        NUM_CLASSES = num_classes if num_classes is not None else (self.num_classes or 0)
        if NUM_CLASSES == 0:
            # Fallback: infer from color_dict
            NUM_CLASSES = max(color_dict.values()) + 1 if color_dict else 0

        all_preds, all_gts = [], []
        print(f"Evaluating {len(pred_files)} images...")
        for pred_file in tqdm(pred_files, desc="Evaluating images"):
            pred_img = Image.open(pred_file).convert("L")
            pred_np = np.array(pred_img)
            gt_file = self.ground_truth_dir / pred_file.name
            if not gt_file.exists():
                # Skip silently (legacy behavior suppressed warnings)
                continue
            gt_rgb = np.array(Image.open(gt_file).convert('RGB'))
            gt_bgr = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
            gt_np = self.image_to_grayscale(gt_bgr, color_dict)
            all_preds.append(torch.tensor(pred_np, dtype=torch.int).flatten())
            all_gts.append(torch.tensor(gt_np, dtype=torch.int).flatten())

        print()
        if not all_preds:
            print("No valid image pairs found for evaluation.")
            print("="*100)
            return None

        all_preds = torch.cat(all_preds)
        all_gts = torch.cat(all_gts)

        pred_unique = torch.unique(all_preds)
        gt_unique = torch.unique(all_gts)
        print(f"Unique values in predictions: {pred_unique.tolist()}")
        print(f"Unique values in ground truth: {gt_unique.tolist()}")
        print(f"Max prediction value: {int(pred_unique.max().item())}")
        print(f"Max ground truth value: {int(gt_unique.max().item())}")

        mpa_metric_per_class = torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES, average='none')
        iou_metric_per_class = torchmetrics.JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average='none')
        m_acc_per_class = mpa_metric_per_class(all_preds, all_gts)
        miou_per_class = iou_metric_per_class(all_preds, all_gts)

        iou_sum = 0.0
        acc_sum = 0.0
        valid_classes = 0
        for cls in range(NUM_CLASSES):
            class_iou = miou_per_class[cls].item()
            class_acc = m_acc_per_class[cls].item()
            print(f"Class {cls} mPA: {class_acc * 100:.2f}, mIoU: {class_iou * 100:.2f}")
            iou_sum += class_iou
            acc_sum += class_acc
            valid_classes += 1
        iou_avg = (iou_sum / valid_classes) if valid_classes else 0.0
        acc_avg = (acc_sum / valid_classes) if valid_classes else 0.0
        print(f"Global mPA: {acc_avg * 100:.2f}%")
        print(f"Global mIoU: {iou_avg * 100:.2f}%")

        # Legacy results file
        eval_results = {
            "global_mpa": float(acc_avg),
            "global_miou": float(iou_avg),
            "per_class_mpa": [float(m_acc_per_class[i]) for i in range(NUM_CLASSES)],
            "per_class_miou": [float(miou_per_class[i]) for i in range(NUM_CLASSES)],
            "num_classes": int(NUM_CLASSES),
            "num_images_evaluated": len(pred_files),
            "used_external_color_dict": color_dict is not None
        }
        legacy_path = self.output_dir / "evaluation_results.json"
        with open(legacy_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nEvaluation results saved to: {legacy_path}")

        # Aggregation JSON expected by run.py
        per_class_iou_dict = {int(i): float(miou_per_class[i]) for i in range(NUM_CLASSES)}
        mean_per_class_iou = float(np.mean(list(per_class_iou_dict.values()))) if per_class_iou_dict else 0.0
        seg_metrics = {
            "global_miou": float(iou_avg),
            "mean_per_class_iou": mean_per_class_iou,
            "per_class_iou": per_class_iou_dict,
            "images_evaluated": len(pred_files),
            "classes_evaluated": int(NUM_CLASSES)
        }
        seg_metrics_path = self.output_dir / "stats" / "segmentation_metrics.json"
        with open(seg_metrics_path, 'w') as f:
            json.dump(seg_metrics, f, indent=2)
        print(f"Segmentation metrics saved to: {seg_metrics_path}")
        print("="*100)
        return seg_metrics

# CLI entrypoints (restored) -- REINSERT after corruption
def main():
    parser = argparse.ArgumentParser(description="Run the Unified AutoLabeler")
    parser.add_argument("--images", required=True, help="Path to the directory containing input images")
    parser.add_argument("--ground-truth", required=False, help="Path to the directory containing ground truth masks (optional for sparseGT-only mode)")
    parser.add_argument("--output", required=True, help="Path to the output directory")
    parser.add_argument("--strategy", required=True,
                        choices=["list", "random", "grid", "SAM2_guided", "dynamicPoints_onlyA", "dynamicPoints", "dynamicPointsLargestGT"],
                        help="Point selection strategy to use")
    parser.add_argument("--num-points", type=int, default=30, help="Number of points to select")
    parser.add_argument("--num-classes", type=int, help="Total number of classes (required if no --color-dict)")
    parser.add_argument("--maskSLIC", action="store_true", help="Use maskSLIC for segmentation")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--visualizations", action="store_true", help="Enable saving visualizations")
    parser.add_argument("--visualization", action="store_true", dest="visualizations", help=argparse.SUPPRESS)
    parser.add_argument("--strategy-kwargs", type=str, default="{}", help="JSON string of strategy-specific parameters")
    parser.add_argument("--points-file", type=str, help="Path to JSON file containing points for 'list' strategy")
    parser.add_argument("--color-dict", type=str, help="Path to external color dictionary JSON file for evaluation")
    parser.add_argument("--downscale-auto", action="store_true", help="Enable automatic heuristic downscaling for auxiliary point proposal data (SAM2 remains full-res)")
    parser.add_argument("--downscale-fixed", type=float, help="Use fixed downscale factor (e.g. 0.5). Overrides --downscale-auto if provided")
    parser.add_argument("--debug-expanded-masks", action="store_true", dest="debug_save_expanded_masks",
                        help="Save each expanded SAM2 mask separately plus overlap diagnostics")
    args = parser.parse_args()

    strategy_kwargs = json.loads(args.strategy_kwargs)
    if args.points_file and args.strategy == "list":
        strategy_kwargs["points_json_path"] = args.points_file

    color_dict = None
    if args.color_dict:
        try:
            print(f"Loading external color dictionary from {args.color_dict}")
            with open(args.color_dict, 'r') as f:
                mapping = json.load(f)
            # Accept either color_to_label nested dict or flat mapping
            if "color_to_label" in mapping:
                color_dict_raw = mapping["color_to_label"]
            else:
                color_dict_raw = mapping
            color_dict = {}
            for k, v in color_dict_raw.items():
                if isinstance(v, int):
                    # key is color string
                    ks = k.strip()
                    ks = ks.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
                    parts = [p.strip() for p in ks.split(',') if p.strip()]
                    if len(parts) == 3:
                        try:
                            tup = tuple(int(p) for p in parts)
                            color_dict[tup] = v
                        except ValueError:
                            continue
            print(f"Loaded external color dictionary with {len(color_dict)} entries")
        except Exception as e:
            print(f"Error loading color dictionary: {e}")
            color_dict = None

    if not color_dict and args.num_classes is None:
        print("INFO: Neither --color-dict nor --num-classes provided; num_classes will be inferred from observed labels. Evaluation will be skipped.")

    if args.ground_truth:
        print("Running in RGB ground truth mode")
    else:
        print("Running in sparseGT-only mode")
        if not color_dict:
            print("WARNING: sparseGT-only mode without color_dict may limit evaluation; ensure points carry class info.")

    # Instantiate AutoLabeler irrespective of the mode so it's always defined
    auto_labeler = AutoLabeler(
        images_dir=args.images,
        ground_truth_dir=args.ground_truth,
        output_dir=args.output,
        save_visualizations=args.visualizations,
        debug_save_expanded_masks=args.debug_save_expanded_masks,
        device=args.device,
        point_selection_strategy=args.strategy,
        num_points=args.num_points,
        use_maskSLIC=args.maskSLIC,
        num_classes=args.num_classes,
        downscale_auto=args.downscale_auto,
        downscale_fixed=args.downscale_fixed,
        **strategy_kwargs
    )
    auto_labeler.run(color_dict)

if __name__ == "__main__":
    main()
