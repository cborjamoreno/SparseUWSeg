import numpy as np
import torch
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.measure import perimeter
from sklearn.cluster import KMeans


class Segmenter:
    def __init__(self, image, checkpoint_path, model_config, device="cuda"):
        """
        Initialize the Segmenter class, generate masks with SAM for the given image.

        Args:
            image (numpy.ndarray): Image as a NumPy array in RGB format.
            checkpoint_path (str): Path to the SAM model checkpoint file.
            model_config (str): Path to the SAM model configuration file.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        # Check that the image is a valid NumPy array
        assert image is not None, "An image must be provided."
        self.image = image  # Store the image directly
        self.height = image.shape[0]
        self.width = image.shape[1]

        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Device setup (use GPU if available)
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"

        # dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dinov2_vitg14.to(device)


        # Load the SAM model
        self.sam_model = self._load_sam_model(checkpoint_path, model_config, self.device)

        # Initialize the SAM mask generator
        self.mask_generator = SAM2AutomaticMaskGenerator(model=self.sam_model,
                                                    points_per_side=64,
                                                    points_per_patch=128,
                                                    pred_iou_threshold=0.7,
                                                    stability_score_thresh=0.92,
                                                    stability_score_offset=0.7,
                                                    crop_n_layers=1,
                                                    box_nms_thresh=0.7,
                                                    )

        # Generate masks for the provided image
        self.masks = self._generate_masks()

        self.selected_masks = set()
        
        self.predictor = SAM2ImagePredictor(self.sam_model)
        self.predictor.set_image(image)

    def _load_sam_model(self, checkpoint_path, model_config, device):
        """
        Load the SAM model.

        Args:
            checkpoint_path (str): Path to the SAM checkpoint.
            model_config (str): Path to the SAM model configuration file.
            device (str): Device to load the model on ("cuda" or "cpu").

        Returns:
            torch.nn.Module: Loaded SAM model.
        """
        print("Loading SAM model...")
        model = build_sam2(
            model_config, checkpoint_path, device=device, apply_postprocessing=False
        )
        model.to(device)
        print("SAM model loaded successfully.")
        return model

    def _generate_masks(self):
        """
        Generate masks for the given image using SAM and filter out non-informative masks.

        Returns:
            list: List of generated masks for the image, excluding non-informative masks.
        """
        print("Generating masks for the provided image...")
        masks, _ = self.mask_generator.generate(self.image)

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
        return filtered_masks

    def _compute_mask_metrics(self, mask, score):
        """
        Compute and normalize mask metrics: compactness, size penalty, and score.

        Args:
            mask (np.array): Binary mask for the segment.
            score (float): Score assigned by SAM for the mask.
            image_shape (tuple): Shape of the image as (height, width).

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
        # Higher compactness for well-defined, continuous masks, lower for scattered/irregular ones
        compactness = min(raw_compactness, 1)  # Ensure compactness doesn't exceed 1

        # Normalized size penalty
        total_pixels = self.height * self.width

        normalized_area = mask_area / total_pixels  # Fraction of the image covered by the mask

        # Gentle penalty for very small masks (e.g., < 1% of image)
        if normalized_area < 0.001:  # Only apply penalty for masks smaller than 1% of the image
            small_mask_penalty = normalized_area ** 4  # Soft quadratic penalty
        else:
            small_mask_penalty = 0  # No penalty for larger masks

        # Large mask penalty (unchanged)
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

    def get_best_point(self):
        """
        Get the best point using a simple approach:
        1. Calculate centroid of each mask
        2. Cluster very close centroids together
        3. Select the centroid of the largest unselected mask
        """
        if not hasattr(self, 'selected_points'):
            self.selected_points = []
        
        # Calculate centroids and areas for all unselected masks
        mask_info = []
        for mask in self.masks:
            if id(mask) not in self.selected_masks:
                segmentation = mask['segmentation']
                indices = list(zip(*segmentation.nonzero()))
                if indices:
                    centroid = np.mean(indices, axis=0)
                    area = mask['area']
                    mask_info.append({
                        'centroid': centroid,
                        'area': area,
                        'mask_id': id(mask)
                    })
        
        if not mask_info:
            return None
            
        # Sort masks by area in descending order
        mask_info.sort(key=lambda x: x['area'], reverse=True)
        
        # Simple clustering of very close centroids
        min_distance = 20  # Minimum distance between centroids to consider them different
        clustered_info = []
        
        for info in mask_info:
            centroid = info['centroid']
            # Check if this centroid is close to any already clustered centroid
            merged = False
            for cluster in clustered_info:
                dist = np.sqrt(np.sum((centroid - cluster['centroid']) ** 2))
                if dist < min_distance:
                    # Merge with existing cluster (weighted average by area)
                    total_area = cluster['area'] + info['area']
                    cluster['centroid'] = (cluster['centroid'] * cluster['area'] + 
                                         centroid * info['area']) / total_area
                    cluster['area'] = total_area
                    cluster['mask_ids'].add(info['mask_id'])
                    merged = True
                    break
            
            if not merged:
                # Create new cluster
                clustered_info.append({
                    'centroid': centroid,
                    'area': info['area'],
                    'mask_ids': {info['mask_id']}
                })
        
        # Find the first cluster that's far enough from all selected points
        min_selection_distance = 50  # Minimum distance from previously selected points
        
        for cluster in clustered_info:
            centroid = cluster['centroid']
            # Check distance to all previously selected points
            too_close = False
            for selected_point in self.selected_points:
                dist = np.sqrt(np.sum((centroid - selected_point) ** 2))
                if dist < min_selection_distance:
                    too_close = True
                    break
            
            if not too_close:
                # Found a good point, update selected masks and points
                self.selected_masks.update(cluster['mask_ids'])
                self.selected_points.append(centroid)
                return tuple(centroid.astype(int))
        
        # If we couldn't find a point far enough from selected points
        return None
    
    def propagate_points(self, points, labels):
        """
        Propagate points into a mask

        Args:
            points: Point prompt coordinates
            lables: Point prompt labels. 1 if positive, 0 if negative.

        Returns:
            np.array: Mask propagated from points.
        """
        
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        selected_mask = self._weighted_mask_selection(masks, scores)

        mask_input = logits[selected_mask, :, :]  # Choose the model's best mask
        # mask_input = logits[np.argmax(scores), :, :]  # Choose the mask with the highest score
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=mask_input[None, :, :],
            multimask_output=True,
        )

        return masks[self._weighted_mask_selection(masks, scores)]


class PointSelector:
    def __init__(self, height, width, grid_size=8):
        """
        Initialize the point selector.
        
        Args:
            height (int): Image height
            width (int): Image width
            grid_size (int): Number of grid cells in each dimension
        """
        self.height = height
        self.width = width
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