import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import cv2

# Add the current directory to the Python path to find the local segment_anything folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from segment_anything.build_sam import build_sam_vit_b
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from segment_anything.predictor import SamPredictor
from skimage.measure import perimeter
from sklearn.cluster import KMeans


class Segmenter:
    def __init__(self, image, sam_checkpoint_path, sam2_checkpoint_path, sam2_config_path, device="cuda"):
        """
        Initialize the Segmenter class, generate masks with SAM2 and use SAM for point prediction.

        Args:
            image (numpy.ndarray): Image as a NumPy array in RGB format.
            sam_checkpoint_path (str): Path to the SAM model checkpoint file for point prediction.
            sam2_checkpoint_path (str): Path to the SAM2 model checkpoint file for automatic mask generation.
            sam2_config_path (str): Path to the SAM2 config file.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        # Check that the image is a valid NumPy array
        assert image is not None, "An image must be provided."
        self.image = image  # Store the image directly
        self.height = image.shape[0]
        self.width = image.shape[1]

        # Device setup (use GPU if available)
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Store checkpoint paths
        self.sam_checkpoint_path = sam_checkpoint_path
        self.sam2_checkpoint_path = sam2_checkpoint_path
        self.sam2_config_path = sam2_config_path
        
        # Initialize models as None - they will be loaded when needed
        self.sam_model = None
        self.sam2_model = None
        self.predictor = None
        self.mask_generator = None
        
        # Initialize expanded areas mask
        self.expanded_areas_mask = np.zeros((self.height, self.width), dtype=bool)
        
        # Generate masks for the provided image
        self.masks = self._generate_masks()
        print(f"Generated {len(self.masks)} masks")

        self.selected_masks = set()
        self.selected_points = []  # Add this line
        self.rejected_masks = set()  # Add this line too for completeness
        
        # Initialize the point selector
        self.point_selector = PointSelector(self.height, self.width, self)

    def _generate_masks(self):
        """
        Generate masks for the given image using SAM2.

        Returns:
            list: List of generated masks for the image, excluding non-informative masks.
        """
        print("Generating masks for the provided image...")
        
        # Load SAM2 model for mask generation
        print("Loading SAM2 model for automatic mask generation...")

        # Create autocast context manager
        self.autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self.autocast_context.__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        from sam2.build_sam import build_sam2
        self.sam2_model = build_sam2(
            self.sam2_config_path, self.sam2_checkpoint_path, device=self.device, apply_postprocessing=False
        )

        self.sam2_model.to(self.device)
        print("SAM2 model loaded successfully.")
        
        # Initialize the SAM2 mask generator
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2_model,
            points_per_side=64,
            points_per_patch=128,
            pred_iou_threshold=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
        )
        
        # Generate masks
        masks = self.mask_generator.generate(self.image)[0]

        # Image area (width x height)
        image_area = self.image.shape[0] * self.image.shape[1]

        # Threshold to determine if a mask covers "too much" of the image as a percentage
        coverage_threshold = 0.95  # Adjust based on your needs (0.95 = 95%)

        # Filter out masks that are non-informative
        filtered_masks = []
        for mask in masks:
            # Calculate mask area
            mask_area = mask['area']  # This should be available in SAM-generated masks

            # If the mask covers less than the threshold, keep it
            if (mask_area / image_area) < coverage_threshold:
                filtered_masks.append(mask)

        print(f"Detected {len(masks)} masks, kept {len(filtered_masks)} after filtering.")
        
        # Exit the autocast context
        if hasattr(self, 'autocast_context'):
            self.autocast_context.__exit__(None, None, None)
            print("Exited autocast context")
        
        # Move SAM2 model to CPU to free GPU memory
        if self.device == "cuda":
            self.sam2_model.to("cpu")
            torch.cuda.empty_cache()
            print("Moved SAM2 model to CPU to free GPU memory")

        # Load SAM model for point propagation if not already loaded
        if self.sam_model is None:
            print("Loading SAM model for point prediction...")
            self.sam_model = build_sam_vit_b(checkpoint=self.sam_checkpoint_path)
            self.sam_model.to(self.device)
            print("SAM model loaded successfully.")
            
            # Initialize SAM predictor for point propagation
            self.predictor = SamPredictor(self.sam_model)
            self.predictor.set_image(self.image)
        
        # Predict expansion for each mask centroid and sort by predicted area
        print("Predicting expansions for mask centroids...")
        mask_info = []
        for mask in filtered_masks:
            segmentation = mask['segmentation']
            indices = list(zip(*segmentation.nonzero()))
            if indices:
                # Calculate centroid in [y, x] format
                centroid = np.mean(indices, axis=0)
                # Convert to [x, y] format for prediction
                point = np.array([[centroid[1], centroid[0]]])
                labels = np.array([1])  # Positive point
                
                # Predict expansion for this centroid
                masks, scores, logits = self.predictor.predict(
                    point_coords=point,
                    point_labels=labels,
                    multimask_output=True,
                )
                
                if masks is not None:
                    # Get the best mask using weighted selection
                    best_mask_idx = self._weighted_mask_selection(masks, scores)
                    predicted_mask = masks[best_mask_idx]
                    predicted_area = predicted_mask.sum()
                    
                    mask_info.append({
                        'mask': mask,
                        'centroid': centroid,  # Keep in [y, x] format for display
                        'point': point[0],     # Store [x, y] format for prediction
                        'area': predicted_area
                    })
        
        # Sort masks by predicted area in descending order
        mask_info.sort(key=lambda x: x['area'], reverse=True)
        
        # Update filtered_masks with sorted order
        filtered_masks = [info['mask'] for info in mask_info]
        print("Sorted masks by predicted expansion area")
            
        return filtered_masks

    def _compute_mask_metrics(self, mask, score):
        """
        Compute and normalize mask metrics: compactness, size penalty, and score.

        Args:
            mask (np.array): Binary mask for the segment.
            score (float): Score assigned by SAM for the mask.

        Returns:
            tuple: Normalized compactness, size penalty, and score.
        """
        # Mask metrics
        mask_area = mask.sum()  # Total pixels in the mask
        mask_perimeter = perimeter(mask)  # Perimeter of the mask

        # Compactness: Avoid divide-by-zero errors
        if mask_area > 0:
            # Ideal perimeter for a circle with the same area
            ideal_perimeter = 2 * np.sqrt(np.pi * mask_area)

            # Compactness: The ratio of the perimeter to the ideal perimeter (closer to 1 is more compact)
            if mask_perimeter > 0:
                raw_compactness = ideal_perimeter / mask_perimeter  # Inverse, so lower perimeter = higher compactness
            else:
                raw_compactness = 0  # Handle the case when mask_perimeter is 0
        else:
            raw_compactness = 0

        # Normalize compactness (keeping compactness between 0 and 1)
        compactness = min(raw_compactness, 1)  # Ensure compactness doesn't exceed 1

        # Normalized size penalty
        total_pixels = self.height * self.width
        normalized_area = mask_area / total_pixels  # Fraction of the image covered by the mask

        # Gentle penalty for very small masks (e.g., < 1% of image)
        if normalized_area < 0.001:  # Only apply penalty for masks smaller than 1% of the image
            small_mask_penalty = normalized_area ** 4  # Soft quadratic penalty
        else:
            small_mask_penalty = 0  # No penalty for larger masks

        # Large mask penalty
        large_mask_penalty = (normalized_area - 0.4) ** 4 if normalized_area > 0.5 else 0

        # Combine penalties gently
        size_penalty = normalized_area + small_mask_penalty + large_mask_penalty

        # Return normalized metrics
        return compactness, size_penalty, score

    def _weighted_mask_selection(self, masks, scores, weights=(1.0, 0.8, 1.4), point=None, label=None):
        best_score = -np.inf
        best_index = -1  # Initialize with an invalid index

        w_s, w_c, w_a = weights  # Weights for SAM Score, Compactness, and Size

        for i, mask in enumerate(masks):
            # Compute metrics
            compactness, size_penalty, sam_score = self._compute_mask_metrics(mask, scores[i])

            # Weighted score (nonlinear terms)
            weighted_score = (
                    w_s * sam_score +  # Higher SAM score is better
                    w_c * np.log(1 + compactness) -  # Higher compactness is better (log smoothing)
                    w_a * size_penalty  # Lower size penalty is better
            )

            # Select best mask
            if weighted_score > best_score:
                best_score = weighted_score
                best_index = i  # Store the index of the best mask

        return best_index

    def get_best_point(self, label_predictor=None, expanded_masks=None):
        """
        Delegate to PointSelector to get the best point for labeling.
        
        Args:
            label_predictor: Optional LabelPredictor with trained prototypes
            expanded_masks: List of already expanded masks to avoid
            
        Returns:
            (row, col) tuple of the suggested point position
        """
        
        return self.point_selector.get_best_point(label_predictor, expanded_masks)
    
    def propagate_points(self, points, labels, update_expanded_mask=True):
        """
        Propagate points into a mask using SAM

        Args:
            points: Point prompt coordinates
            labels: Point prompt labels. 1 if positive, 0 if negative.
            update_expanded_mask (bool): Whether to update the expanded_areas_mask. 
                                       Should be True only for actual point predictions, 
                                       False for dynamic expansion visualization.

        Returns:
            np.array: Mask propagated from points.
        """
        
        # Convert points and labels to the correct format
        # The predictor expects NumPy arrays, not PyTorch tensors
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        elif isinstance(points, list):
            points = np.array(points)

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)
        
        # Ensure points and labels are in the correct shape
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        if len(labels.shape) == 0:
            labels = labels.reshape(1)

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        predicted_mask = masks[self._weighted_mask_selection(masks, scores)]
        
        # Only update the expanded areas mask if this is an actual point prediction
        if update_expanded_mask:
            self.expanded_areas_mask = np.logical_or(self.expanded_areas_mask, predicted_mask)
            print(f"Updated expanded areas mask. New total area: {np.sum(self.expanded_areas_mask)} pixels")
        
        return predicted_mask
    
    def cleanup(self):
        """
        Clean up resources by moving models to CPU and clearing GPU memory.
        This should be called when switching to a new image.
        """
        if self.device == "cuda":
            # Move SAM model to CPU if it exists
            if self.sam_model is not None:
                self.sam_model.to("cpu")
                print("Moved SAM model to CPU")
            
            # Move SAM2 model to CPU if it exists
            if self.sam2_model is not None:
                self.sam2_model.to("cpu")
                print("Moved SAM2 model to CPU")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            print("Cleared GPU memory")

    def initialize_sam_predictor(self):
        """Initialize the SAM predictor after generating automatic masks"""
        if self.sam_model is None:
            self.sam_model = build_sam_vit_b(checkpoint=self.sam_checkpoint_path)
            self.sam_model.to(self.device)
            self.sam_model.eval()
            self.predictor = SamPredictor(self.sam_model)
            self.predictor.set_image(self.image)


class PointSelector:
    def __init__(self, height, width, segmenter, grid_size=8):
        """
        Initialize the point selector.
        
        Args:
            height (int): Image height
            width (int): Image width
            segmenter: Reference to the parent segmenter
            grid_size (int): Number of grid cells in each dimension
        """
        self.height = height
        self.width = width
        self.segmenter = segmenter
        self.grid_size = grid_size
        self.grid_h = height // grid_size
        self.grid_w = width // grid_size
        self.selected_points = []  # Track selected points
        self.selected_masks = set()  # Track selected mask IDs
        
    def update_selection_state(self, point, mask_id):
        """
        Update the selection state with a new point and its associated mask.
        
        Args:
            point (tuple): (row, col) coordinates of selected point
            mask_id (int): ID of the selected mask
        """
        self.selected_points.append(point)
        self.selected_masks.add(mask_id)
        
    def _calculate_coverage_score(self, point):
        """
        Calculate how well a point covers the image based on existing selections.
        
        Args:
            point (tuple): (row, col) coordinates of candidate point
            
        Returns:
            float: Coverage score (lower is better)
        """
        if not self.selected_points:
            return 0.0
            
        # Calculate distance to nearest selected point
        min_dist = float('inf')
        for selected_point in self.selected_points:
            dist = np.sqrt((point[0] - selected_point[0])**2 + 
                         (point[1] - selected_point[1])**2)
            min_dist = min(min_dist, dist)
            
        # Normalize distance by image diagonal
        max_dist = np.sqrt(self.height**2 + self.width**2)
        return min_dist / max_dist
        
    def generate_candidates(self, image, masks, distance_threshold=50, visualize=True):
        """
        Generate and combine all candidate points with adaptive recalculation.
        
        Args:
            image: Original RGB image
            masks: List of SAM masks
            distance_threshold: Minimum distance between points
            visualize: Whether to show the visualization
            
        Returns:
            np.array: Array of final candidate points
        """
        # Generate points from both sources
        grid_points = self.generate_grid_points()
        mask_points = self.generate_mask_points(masks)
        
        # Combine all points
        all_points = np.vstack([grid_points, mask_points]) if len(mask_points) > 0 else grid_points
        
        # Calculate coverage scores for each point
        coverage_scores = np.array([self._calculate_coverage_score(point) for point in all_points])
        
        # Adjust clustering based on coverage
        if len(self.selected_points) > 0:
            # If we have selected points, be more conservative with clustering
            distance_threshold = min(150, distance_threshold * (1 + len(self.selected_points) * 0.1))
        
        # Cluster points to remove redundancy
        final_points = self.cluster_points(all_points, distance_threshold=distance_threshold)
        
        # Sort final points by coverage score
        final_scores = np.array([self._calculate_coverage_score(point) for point in final_points])
        sorted_indices = np.argsort(final_scores)
        final_points = final_points[sorted_indices]
        
        if visualize:
            self.visualize_points(image, grid_points, mask_points, final_points)
        
        return final_points

    def generate_grid_points(self):
        """
        Generate candidate points based on grid centers.
        
        Returns:
            np.array: Array of (row, col) coordinates for grid centers
        """
        grid_points = []
        
        # Generate points at the center of each grid cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate center of current grid cell
                center_y = int((i + 0.5) * self.grid_h)
                center_x = int((j + 0.5) * self.grid_w)
                
                # Ensure points are within image bounds
                center_y = min(center_y, self.height - 1)
                center_x = min(center_x, self.width - 1)
                
                grid_points.append([center_y, center_x])
                
        return np.array(grid_points)
    
    def generate_mask_points(self, masks):
        """
        Generate candidate points from mask centroids.
        
        Args:
            masks (list): List of SAM masks, each with a 'segmentation' key
            
        Returns:
            np.array: Array of (row, col) coordinates for mask centroids
        """
        mask_points = []
        
        for mask in masks:
            segmentation = mask['segmentation']
            indices = list(zip(*segmentation.nonzero()))
            
            if indices:
                # Calculate centroid
                centroid = np.mean(indices, axis=0)
                mask_points.append([centroid[0], centroid[1]])
        
        return np.array(mask_points)
    
    def cluster_points(self, points, n_clusters=None, distance_threshold=50):
        """
        Cluster points to remove redundancy while preserving important centroids.
        
        Args:
            points (np.array): Array of (row, col) coordinates
            n_clusters (int, optional): Number of clusters. If None, calculated based on distance_threshold
            distance_threshold (float): Minimum distance between points
            
        Returns:
            np.array: Array of clustered points
        """
        if len(points) == 0:
            return np.array([])
            
        # Increase the distance threshold to be more conservative
        distance_threshold = 100  # Increased from 50 to 100
        
        if n_clusters is None:
            # Estimate number of clusters based on image size and distance threshold
            image_diagonal = np.sqrt(self.height**2 + self.width**2)
            n_clusters = max(1, int(image_diagonal / distance_threshold))
            n_clusters = min(n_clusters, len(points))
        
        # Perform clustering with more clusters to preserve more points
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(points)
        
        # Get cluster centers
        clustered_points = kmeans.cluster_centers_
        
        # Round to integer coordinates
        clustered_points = np.round(clustered_points).astype(int)
        
        # Ensure points are within image bounds
        clustered_points[:, 0] = np.clip(clustered_points[:, 0], 0, self.height - 1)
        clustered_points[:, 1] = np.clip(clustered_points[:, 1], 0, self.width - 1)
        
        return clustered_points
    
    def visualize_points(self, image, grid_points=None, mask_points=None, final_points=None):
        """
        Visualize different stages of point generation on the image.
        
        Args:
            image: Original RGB image
            grid_points: Points generated from grid
            mask_points: Points generated from mask centroids
            final_points: Final clustered points
        """
        # Create a figure with subplots based on what we want to show
        n_plots = 1 + (grid_points is not None) + (mask_points is not None) + (final_points is not None)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0
        
        # Original image
        axes[plot_idx].imshow(image)
        axes[plot_idx].set_title('Original Image')
        axes[plot_idx].axis('off')
        
        # Grid points
        if grid_points is not None:
            plot_idx += 1
            axes[plot_idx].imshow(image)
            axes[plot_idx].scatter(grid_points[:, 1], grid_points[:, 0], 
                                 c='yellow', marker='+', s=100, label='Grid Points')
            axes[plot_idx].set_title('Grid-based Points')
            axes[plot_idx].axis('off')
            axes[plot_idx].legend()

        # Mask centroid points
        if mask_points is not None:
            plot_idx += 1
            axes[plot_idx].imshow(image)
            axes[plot_idx].scatter(mask_points[:, 1], mask_points[:, 0], 
                                 c='red', marker='x', s=100, label='Mask Centroids')
            axes[plot_idx].set_title('Mask Centroids')
            axes[plot_idx].axis('off')
            axes[plot_idx].legend()

        # Final clustered points
        if final_points is not None:
            plot_idx += 1
            axes[plot_idx].imshow(image)
            axes[plot_idx].scatter(final_points[:, 1], final_points[:, 0], 
                                 c='green', marker='*', s=200, label='Final Points')
            axes[plot_idx].set_title('Final Clustered Points')
            axes[plot_idx].axis('off')
            axes[plot_idx].legend()

        plt.tight_layout()
        plt.show()
    
    def get_best_point(self, label_predictor=None, expanded_masks=None):
        """
        Get the best point for labeling, prioritizing:
        1. Least similar to known classes (encouraging new class discovery)
        2. Significant area (favoring larger objects)
        3. Not overlapping with already expanded masks
        
        Args:
            label_predictor: Optional LabelPredictor with trained prototypes
            expanded_masks: List of already expanded masks to avoid
            
        Returns:
            (row, col) tuple of the suggested point position
        """
        # Define constants
        COVERAGE_THRESHOLD = 0.3
        DISTANCE_THRESHOLD = 5
        
        # Check if we can use contrastive learning
        use_contrastive = (label_predictor is not None and 
                           hasattr(label_predictor, 'prototypes') and 
                           label_predictor.prototypes)
        
        if use_contrastive:
            print("\n===== INTELLIGENT POINT SUGGESTION =====")
            print("Using contrastive learning to find dissimilar objects...")
        
        # Collect candidate masks with scores
        candidates = []
        
        # Process each mask
        for mask in self.segmenter.masks:
            # Skip if mask is already selected or rejected
            if id(mask) in self.segmenter.selected_masks or id(mask) in self.segmenter.rejected_masks:
                continue
                
            segmentation = mask['segmentation']
            indices = list(zip(*segmentation.nonzero()))
            if not indices:
                continue
                
            # Calculate centroid in [y, x] format for display
            centroid = np.mean(indices, axis=0)
            cent_y, cent_x = int(centroid[0]), int(centroid[1])
            
            # NEW CODE: Direct check if centroid falls within any existing mask
            if expanded_masks:
                centroid_in_mask = False
                for i, (mask_obj, label, _) in enumerate(expanded_masks):
                    # Double-check bounds
                    if 0 <= cent_y < mask_obj.shape[0] and 0 <= cent_x < mask_obj.shape[1]:
                        if mask_obj[cent_y, cent_x]:
                            print(f"Point {(cent_y, cent_x)} overlaps mask #{i} with label '{label}'")
                            centroid_in_mask = True
                            break
            
                if centroid_in_mask:
                    self.segmenter.rejected_masks.add(id(mask))
                    continue
            
            # Skip if centroid is too close to any already selected point
            is_near_selected = False
            for selected_point in self.segmenter.selected_points:
                distance = np.sqrt(np.sum((centroid - selected_point) ** 2))
                if distance <= DISTANCE_THRESHOLD:
                    is_near_selected = True
                    break
            
            if is_near_selected:
                self.segmenter.rejected_masks.add(id(mask))
                continue
            
            # Convert to [x, y] format for prediction
            point = np.array([[centroid[1], centroid[0]]])
            labels = np.array([1])
            
            # Predict expansion for this centroid
            masks, scores, logits = self.segmenter.predictor.predict(
                point_coords=point,
                point_labels=labels,
                multimask_output=True,
            )
            
            if masks is None:
                continue
                
            best_mask_idx = self.segmenter._weighted_mask_selection(masks, scores)
            predicted_mask = masks[best_mask_idx]
            
            # Check overlap with expanded areas
            new_pixels = np.sum(predicted_mask)
            if new_pixels == 0:
                continue
                
            if expanded_masks:
                # Check overlap with provided expanded_masks
                is_overlapping = False
                for mask_obj, _, _ in expanded_masks:
                    overlap = np.logical_and(predicted_mask, mask_obj)
                    overlap_area = np.sum(overlap)
                    if overlap_area / new_pixels >= COVERAGE_THRESHOLD:
                        is_overlapping = True
                        break
                
                if is_overlapping:
                    self.segmenter.rejected_masks.add(id(mask))
                    continue
            else:
                # Use the original expanded_areas_mask if available
                if hasattr(self.segmenter, 'expanded_areas_mask') and self.segmenter.expanded_areas_mask is not None:
                    covered_pixels = np.sum(np.logical_and(predicted_mask, self.segmenter.expanded_areas_mask))
                    coverage_ratio = covered_pixels / new_pixels
                    
                    if coverage_ratio >= COVERAGE_THRESHOLD:
                        self.segmenter.rejected_masks.add(id(mask))
                        continue
            
            # First, add absolute size filtering before any scoring
            mask_area = np.sum(predicted_mask)
            min_pixels = 500  # Minimum size threshold
            if mask_area < min_pixels:
                # Skip tiny masks entirely
                self.segmenter.rejected_masks.add(id(mask))
                continue

            # Now add contrastive learning logic
            if use_contrastive:
                # Extract features and embedding for this mask
                try:
                    features = label_predictor._extract_features(self.segmenter.image, predicted_mask)
                    embedding = label_predictor._compute_mask_embedding(features, predicted_mask)
                    
                    if embedding is not None:
                        # Calculate similarity to every known class
                        similarities = []
                        for label, prototype in label_predictor.prototypes.items():
                            try:
                                from sklearn.metrics.pairwise import cosine_similarity
                                sim = cosine_similarity(
                                    np.array(embedding).reshape(1, -1),
                                    np.array(prototype).reshape(1, -1)
                                )[0][0]
                                similarities.append((label, sim))
                            except Exception as e:
                                continue
                        
                        if similarities:
                            # Sort similarities (highest first)
                            similarities.sort(key=lambda x: x[1], reverse=True)
                            similarity = similarities[0][1]
                            closest_class = similarities[0][0]
                            
                            # Calculate dissimilarity score (higher is more different)
                            dissimilarity = 1.0 - max(0.0, similarity)
                            
                            # Calculate area score (higher is larger)
                            # Then change how normalized_area is calculated
                            # Use a sigmoid-like function that heavily penalizes small areas
                            # and rewards masks that are a reasonable size (not too small, not too big)
                            normalized_area = min(1.0, mask_area / (self.height * self.width * 0.05))

                            # For very small masks, apply an exponential penalty
                            if normalized_area < 0.01:  # Masks smaller than 1% of "reasonable size"
                                normalized_area *= normalized_area  # Square it to heavily penalize tiny masks

                            # Adjust the score weighting to balance size and dissimilarity
                            combined_score = 0.5 * dissimilarity + 0.5 * normalized_area  # Equal weighting
                            
                            # Add to candidates
                            candidates.append({
                                'point': tuple(centroid.astype(int)),
                                'mask_id': id(mask),
                                'score': combined_score,
                                'dissimilarity': dissimilarity,
                                'closest_class': closest_class,
                                'similarity': similarity,
                                'area': normalized_area
                            })
                            continue  # Skip the default scoring
                except Exception as e:
                    print(f"Error in contrastive scoring: {e}")
            
            # Default scoring if contrastive learning is not available
            candidates.append({
                'point': tuple(centroid.astype(int)),
                'mask_id': id(mask),
                'score': 0.5,  # Default middle score
                'dissimilarity': 0.5,
                'closest_class': None,
                'similarity': 0.0,
                'area': min(1.0, new_pixels / (self.height * self.width * 0.2))
            })
        
        # If no candidates, return center point
        if not candidates:
            print("No suitable candidates found, using default point")
            return (self.height // 2, self.width // 2)
        
        # Sort by combined score (highest first)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Print top candidates if using contrastive learning
        if use_contrastive:
            print("\nTop suggestions:")
            for i, cand in enumerate(candidates[:3]):
                if i >= len(candidates):
                    break
                print(f"{i+1}. Point: {cand['point']} | " +
                     f"Score: {cand['score']:.3f} | " +
                     f"Dissimilarity: {cand['dissimilarity']:.3f}" +
                     (f" | Closest class: '{cand['closest_class']}' ({cand['similarity']:.3f})" 
                      if cand['closest_class'] else "") +
                     f" | Area: {cand['area']:.3f}")
            
            print(f"Selected point: {candidates[0]['point']}")
            print("========================================\n")
        
        # Mark the best candidate as selected
        best_candidate = candidates[0]
        point_yx = best_candidate['point']  # Already in [y, x] format
        
        # Store in segmenter for future reference
        self.segmenter.selected_masks.add(best_candidate['mask_id'])
        self.segmenter.selected_points.append(np.array(point_yx))
        
        # Debug print to verify coordinates
        print(f"Selected point [y={point_yx[0]}, x={point_yx[1]}] (row, col)")
        
        if candidates and use_contrastive:
            # Debug visualization for top candidate
            best_candidate = candidates[0]
            best_mask_id = best_candidate['mask_id']
            
            # Find the corresponding mask object
            for mask in self.segmenter.masks:
                if id(mask) == best_mask_id:
                    # Get mask segmentation and show debug visualization
                    segmentation = mask['segmentation']
                    point_yx = best_candidate['point']
                    print(f"Showing visualization of SUGGESTED POINT: {point_yx}")
                    self.debug_show_candidate_mask(
                        self.segmenter.image, 
                        segmentation, 
                        point_yx,
                        title=f"SUGGESTED POINT [y={point_yx[0]}, x={point_yx[1]}]"
                    )
                    break
        
        return point_yx  # Return in [y, x] format

    def debug_show_candidate_mask(self, image, mask, centroid, title=None):
        """Save debug visualization to file instead of displaying it."""
        import matplotlib.pyplot as plt
        import os
        import time
        
        # Create output directory
        os.makedirs("debug_plots", exist_ok=True)
        
        # Create a figure with three subplots for better debugging
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Generate a filename
        y, x = int(centroid[0]), int(centroid[1])
        timestamp = int(time.time())
        filename = f"point_y{y}_x{x}_{timestamp}.png"
        filepath = os.path.join("debug_plots", filename)
        
        # Use custom title if provided
        if title:
            plt.suptitle(title, fontsize=16, color='blue')
        else:
            # Check if the centroid is in the mask
            is_in_mask = mask[y, x] > 0 if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] else False
            plt.suptitle(f"Centroid in mask: {'YES' if is_in_mask else 'NO'}", 
                       fontsize=16, color='green' if is_in_mask else 'red')
        
        # 1. Original image with centroid
        axes[0].imshow(image)
        axes[0].scatter(x, y, c='red', marker='+', s=100)
        axes[0].set_title(f'Centroid at [y={y}, x={x}]')
        axes[0].axis('off')
        
        # 2. Binary mask only
        axes[1].imshow(mask, cmap='gray')
        axes[1].scatter(x, y, c='yellow', marker='+', s=100)
        axes[1].set_title('Binary Mask')
        axes[1].axis('off')
        
        # 3. Mask overlay with centroid
        mask_overlay = image.copy()
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask > 0] = [255, 0, 0]  # Red for the mask
        mask_overlay = cv2.addWeighted(mask_overlay, 1.0, mask_rgb, 0.5, 0)
        
        axes[2].imshow(mask_overlay)
        axes[2].scatter(x, y, c='yellow', marker='+', s=100)
        axes[2].set_title('Mask Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save to file instead of showing
        plt.savefig(filepath)
        plt.close(fig)
        
        print(f"Debug plot saved to: {filepath}")
        
        # If the centroid is not in the mask, save a second plot with the true centroid
        is_in_mask = mask[y, x] > 0 if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] else False
        if not is_in_mask:
            indices = list(zip(*mask.nonzero()))
            if indices:
                true_centroid = np.mean(indices, axis=0)
                true_y, true_x = int(true_centroid[0]), int(true_centroid[1])
                
                print(f"WARNING: Provided centroid {(y, x)} is not in the mask!")
                print(f"True mask centroid would be at {(true_y, true_x)}")
                
                # Create a new figure showing both centroids
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(mask_overlay)
                ax.scatter(x, y, c='red', marker='x', s=150, label='Original')
                ax.scatter(true_x, true_y, c='lime', marker='+', s=150, label='True centroid')
                ax.set_title('Corrected Centroid')
                ax.legend()
                ax.axis('off')
                
                corrected_filepath = os.path.join("debug_plots", f"corrected_{filename}")
                plt.savefig(corrected_filepath)
                plt.close(fig)
                
                print(f"Corrected centroid plot saved to: {corrected_filepath}")