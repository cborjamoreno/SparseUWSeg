import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image
import itertools
import math
from sklearn.metrics.pairwise import cosine_similarity

def contrastive_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    # Safe normalization with epsilon
    eps = 1e-8
    anchor = F.normalize(anchor, p=2, dim=0, eps=eps)
    positive = F.normalize(positive, p=2, dim=0, eps=eps)
    negative = F.normalize(negative, p=2, dim=0, eps=eps)
    
    # Check for NaN values after normalization
    if torch.isnan(anchor).any() or torch.isnan(positive).any() or torch.isnan(negative).any():
        print("Warning: NaN values detected after normalization")
        # Print magnitude of each vector before normalization
        print(f"  Magnitudes - Anchor: {torch.norm(anchor):e}, Positive: {torch.norm(positive):e}, Negative: {torch.norm(negative):e}")
        return torch.tensor(0.01)  # Return small non-zero loss to allow training to continue
    
    # Compute distances
    positive_distance = F.pairwise_distance(anchor, positive, p=2)
    negative_distance = F.pairwise_distance(anchor, negative, p=2)
    
    # Protect against NaN in loss calculation
    base_loss = positive_distance - negative_distance + margin
    loss = torch.log(1 + torch.exp(base_loss))
    
    # Print details for debugging
    # print(f"  Positive dist: {positive_distance.item():.4f}, Negative dist: {negative_distance.item():.4f}")
    # print(f"  Base loss: {base_loss.item():.4f}, Final loss: {loss.item():.4f}")
    
    return loss.mean()

# Center padding class for consistent feature extraction
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple=14):  # DINOv2 patch size is 14
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def forward(self, x):
        pads = list(itertools.chain.from_iterable(
            self._get_pad(m) for m in x.shape[-2:]))
        return F.pad(x, pads)

class LabelPredictor:
    def __init__(self, confidence_threshold: float = 0.85, min_examples_per_class: int = 1):
        """Initialize the LabelPredictor with DINOv2 model and feature database."""
        self.confidence_threshold = confidence_threshold
        self.min_examples_per_class = min_examples_per_class
        
        # Initialize database
        self.feature_database: Dict[str, List[torch.Tensor]] = {}
        self.prototypes: Dict[str, torch.Tensor] = {}
        self.last_mask_features = None
        self.last_mask_label = None
        self.spatial_features = None
        self.mask_features = {}
        self.debug_info = {}
        
        # Initialize DINOv2 model
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.patch_size = 14
            
            # Define transform
            self.transform = T.Compose([
                T.ToTensor(),
                CenterPadding(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        except Exception as e:
            print(f"Error initializing DINOv2 model: {e}")
            raise

    def _extract_features(self, image: np.ndarray, *args) -> np.ndarray:
        """Extract spatial features map from the entire image."""
        try:
            # Convert to PIL Image and get dimensions
            pil_image = Image.fromarray(image)
            width, height = pil_image.size
            
            # Calculate padding to ensure dimensions are multiples of patch size
            padded_height = ((height + self.patch_size - 1) // self.patch_size) * self.patch_size
            padded_width = ((width + self.patch_size - 1) // self.patch_size) * self.patch_size
            
            # Create a new padded image
            padded_image = Image.new('RGB', (padded_width, padded_height), (0, 0, 0))
            # Paste the original image centered in the padded image
            paste_x = (padded_width - width) // 2
            paste_y = (padded_height - height) // 2
            padded_image.paste(pil_image, (paste_x, paste_y))
            
            # Apply transformations and run through model
            with torch.inference_mode():
                # Use standard transforms without the CenterPadding since we manually padded
                transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
                
                img_tensor = transform(padded_image).unsqueeze(0).to(self.device)
                features_out = self.model.get_intermediate_layers(img_tensor, n=[23], reshape=True)
                features = torch.cat(features_out, dim=1)[0].cpu().detach()
                
                # Upsample to match original image dimensions
                upsampled = F.interpolate(
                    features.unsqueeze(0), 
                    size=(image.shape[0], image.shape[1]), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                return upsampled.squeeze(0).numpy()
        except Exception as e:
            print(f"Feature extraction error: {e}")
            print(f"Image shape: {image.shape}, expected multiple of {self.patch_size}")
            return np.random.randn(384, image.shape[0], image.shape[1])

    def _extract_spatial_features(self, image: np.ndarray) -> np.ndarray:
        """Extract spatial features from an entire image."""
        # Reuse the improved _extract_features method
        return self._extract_features(image)

    def _compute_mask_embedding(self, spatial_features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute mean embedding for a mask from spatial features."""
        mask_coords = np.argwhere(mask > 0)
        if mask_coords.size == 0:
            return None
        
        mask_features = spatial_features[:, mask_coords[:, 0], mask_coords[:, 1]]
        return np.mean(mask_features, axis=1)
    
    def add_example(self, image: np.ndarray, mask: np.ndarray, label: str) -> None:
        """Add a new labeled example to the feature database."""
        # Extract spatial features if needed
        if self.spatial_features is None:
            self.spatial_features = self._extract_features(image)  # Ensure correct invocation
        
        # Compute mask embedding
        mask_embedding = self._compute_mask_embedding(self.spatial_features, mask)
        
        # Debug similarity information before assigning the label
        print("\n[DEBUG] Similarity information before assigning label:")
        self.should_auto_assign(image, mask)
        
        # Compare with the last mask
        last_comparison = self.compare_with_last_mask(image, mask)
        if last_comparison:
            last_label, last_similarity = last_comparison
            print("\n[DEBUG] Comparison with last mask:")
            print(f"Last mask label: {last_label}")
            print(f"Similarity: {last_similarity:+.4f}")
        
        # Store mask embedding
        self.last_mask_label = label
        if label not in self.feature_database:
            self.feature_database[label] = []
        
        if mask_embedding is not None:
            self.feature_database[label].append(mask_embedding)
            self.mask_features[(label, len(self.feature_database[label]) - 1)] = mask_embedding
        
        # Update prototype
        self._update_prototype(label)


    def _update_prototype(self, label: str) -> None:
        """Update the prototype vector for a label using contrastive learning."""
        if label in self.feature_database and len(self.feature_database[label]) > 1:
            embeddings = torch.stack([torch.tensor(e) for e in self.feature_database[label]])
            anchor = embeddings[0]  # Use the first embedding as the anchor
            positive = embeddings[1]  # Use the second embedding as the positive example
            for other_label, other_embeddings in self.feature_database.items():
                if other_label != label and len(other_embeddings) > 0:
                    negative = torch.tensor(other_embeddings[0])  # Use the first embedding of a different label
                    # Compute contrastive loss
                    loss = contrastive_loss(anchor, positive, negative)
                    print(f"Contrastive loss for label '{label}' vs '{other_label}': {loss.item():.4f}")
            # Update prototype as the mean of embeddings
            self.prototypes[label] = torch.mean(embeddings, dim=0).numpy()

    def predict_label(self, image: np.ndarray, mask: np.ndarray, 
                     top_k: int = 3, force_prediction: bool = True) -> List[Tuple[str, float]]:
        """Predict label for a new mask."""
        if not self.prototypes:
            return []
        
        # Extract features if needed
        if self.spatial_features is None:
            self.spatial_features = self._extract_features(image)
        
        # Compute mask embedding
        new_mask_embedding = self._compute_mask_embedding(self.spatial_features, mask)
        if new_mask_embedding is None:
            return []
        
        # Compute similarities
        similarities = {}
        for label, prototype in self.prototypes.items():
            if force_prediction or len(self.feature_database[label]) >= self.min_examples_per_class:
                try:
                    # Compute similarity with prototype
                    similarity = cosine_similarity(
                        new_mask_embedding.reshape(1, -1),
                        prototype.reshape(1, -1)
                    )[0][0]
                    
                    similarities[label] = similarity
                except Exception as e:
                    print(f"Error calculating similarity for {label}: {e}")
        
        if not similarities:
            return []
        
        # Return top-k predictions
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_similarities[:min(top_k, len(sorted_similarities))]

    def compare_with_last_mask(self, image: np.ndarray, mask: np.ndarray) -> Optional[Tuple[str, float]]:
        """Compare current mask with the last expanded mask using spatial features."""
        if self.last_mask_label is None:
            return None
            
        # Extract spatial features if needed
        if self.spatial_features is None:
            self.spatial_features = self._extract_features(image)  # Ensure correct invocation
        
        # Compute embeddings for current mask
        current_mask_embedding = self._compute_mask_embedding(self.spatial_features, mask)
        
        # Get embeddings for last mask (should be stored when mask is added)
        last_mask_embedding = self.mask_features.get((self.last_mask_label, 0))
        
        if current_mask_embedding is None or last_mask_embedding is None:
            return None
        
        try:
            # Calculate cosine similarity between mask embeddings
            similarity = cosine_similarity(
                current_mask_embedding.reshape(1, -1),
                last_mask_embedding.reshape(1, -1)
            )[0][0]
            
            return (self.last_mask_label, similarity)
        except Exception as e:
            print(f"Error comparing with last mask: {e}")
            return None

    def should_auto_assign(self, image: np.ndarray, mask: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Determine if a label should be automatically assigned."""
        # Get predictions
        predictions = self.predict_label(image, mask, top_k=len(self.prototypes), force_prediction=True)
        if not predictions:
            return False, None
        
        # Get top prediction
        top_label, top_similarity = predictions[0]
        
        # Show similarity table
        print("\n===== SIMILARITY TO CLASS PROTOTYPES =====")
        print(f"{'Class':<15} | {'Spatial':<10} | {'Visual'}")
        print("-" * 50)
        
        for label, spatial_similarity in predictions:
            # Create bar visualization
            bar_length = int((spatial_similarity + 1) * 10)
            bar_length = max(0, min(20, bar_length))
            similarity_bar = "█" * bar_length + "░" * (20 - bar_length)
            highlight = " ← BEST MATCH" if label == top_label else ""
            
            print(f"{label:<15} | {spatial_similarity:+.4f}  | {similarity_bar} {highlight}")
        
        # Compare with last mask
        last_comparison = self.compare_with_last_mask(image, mask)
        if last_comparison:
            last_label, last_similarity = last_comparison
            print("\n===== SIMILARITY TO PREVIOUS MASK =====")
            print(f"{'Previous Label':<15} | {'Similarity':<10} | {'Visual'}")
            print("-" * 50)
            
            # Create bar visualization
            bar_length = int((last_similarity + 1) * 10)
            bar_length = max(0, min(20, bar_length))
            similarity_bar = "█" * bar_length + "░" * (20 - bar_length)
            
            print(f"{last_label:<15} | {last_similarity:+.4f}  | {similarity_bar}")
            print("-" * 50)
            
            # Interpretation
            print("Interpretation: ", end="")
            if last_similarity > 0.8:
                print("Very similar to previous mask")
            elif last_similarity > 0.5:
                print("Quite similar to previous mask")
            elif last_similarity > 0.2:
                print("Moderately similar to previous mask")
            elif last_similarity > 0:
                print("Slightly similar to previous mask")
            else:
                print("Different from previous mask")
        
        # Auto assign?
        print(f"\nThreshold: {self.confidence_threshold:.4f}")
        print(f"Auto-assign: {'Yes' if top_similarity >= self.confidence_threshold else 'No'}")
        print("=======================================")
        
        return top_similarity >= self.confidence_threshold, top_label

    def print_database_status(self, detailed: bool = False) -> None:
        """Print database status."""
        print("\n===== FEATURE DATABASE STATUS =====")
        print(f"Total classes: {len(self.feature_database)}")
        for label, features in self.feature_database.items():
            print(f"Class: '{label}'")
            print(f"  - Examples: {len(features)}")
            if detailed and len(features) > 0:
                print(f"  - Feature shape: {features[0].shape}")
        print("==================================\n")

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the feature database."""
        return {label: len(features) for label, features in self.feature_database.items()}

    def train_contrastive(self, epochs: int = 10, margin: float = 1.0) -> None:
        """Train embeddings using contrastive loss with centrality-based weights."""
        # Keep track of original prototypes to measure improvement
        original_prototypes = {label: proto.copy() if isinstance(proto, np.ndarray) else proto.clone() 
                              for label, proto in self.prototypes.items()}
        
        # Base weights that will be adjusted by centrality
        base_anchor_weight = 0.01
        base_pos_weight = 0.003
        base_neg_weight = 0.01
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_loss = 0.0
            triplet_count = 0
            
            # Iterate through all possible triplets
            for label, embeddings in self.feature_database.items():
                if len(embeddings) < 2:
                    continue  # Skip labels with fewer than 2 examples
                
                # Create tensor versions of all embeddings for this label
                label_embeddings = [torch.tensor(e, dtype=torch.float32) for e in embeddings]
                
                # Get class prototype for centrality calculation
                if label in self.prototypes:
                    class_prototype = torch.tensor(self.prototypes[label], dtype=torch.float32)
                else:
                    # If no prototype exists yet, use mean of available embeddings
                    class_prototype = torch.mean(torch.stack(label_embeddings), dim=0)
                
                # For each anchor-positive pair from the same class
                for i in range(len(label_embeddings)):
                    for j in range(len(label_embeddings)):
                        if i == j:
                            continue  # Skip using the same embedding as both anchor and positive
                        
                        anchor = label_embeddings[i]
                        positive = label_embeddings[j]
                        
                        # Calculate centrality (how central each point is to its class)
                        anchor_norm = F.normalize(anchor, p=2, dim=0)
                        positive_norm = F.normalize(positive, p=2, dim=0)
                        prototype_norm = F.normalize(class_prototype, p=2, dim=0)
                        
                        # Higher value means more central to class
                        anchor_centrality = torch.dot(anchor_norm, prototype_norm).item()
                        positive_centrality = torch.dot(positive_norm, prototype_norm).item()
                        
                        # Adjust weights based on centrality - central points move less
                        # Map from [-1,1] to [0.5,2.0] range for weight scaling
                        anchor_factor = 1.5 * (1.0 - anchor_centrality) + 0.5
                        positive_factor = 1.5 * (1.0 - positive_centrality) + 0.5
                        
                        # Calculate final weights
                        anchor_weight = base_anchor_weight * anchor_factor
                        pos_weight = base_pos_weight * positive_factor
                        
                        # Find negative examples from other classes
                        for neg_label, neg_embeddings in self.feature_database.items():
                            if neg_label == label or not neg_embeddings:
                                continue
                            
                            negative = torch.tensor(neg_embeddings[0], dtype=torch.float32)
                            
                            # Calculate negative centrality to its own class
                            if neg_label in self.prototypes:
                                neg_prototype = torch.tensor(self.prototypes[neg_label], dtype=torch.float32)
                                neg_norm = F.normalize(negative, p=2, dim=0)
                                neg_prototype_norm = F.normalize(neg_prototype, p=2, dim=0)
                                neg_centrality = torch.dot(neg_norm, neg_prototype_norm).item()
                                # Outliers in negative class should move more
                                neg_factor = 1.5 * (1.0 - neg_centrality) + 0.5
                            else:
                                neg_factor = 1.0  # Default factor if no prototype
                            
                            neg_weight = base_neg_weight * neg_factor
                            
                            # print(f"Training triplet: '{label}' pair vs '{neg_label}'")
                            # print(f"  Centrality factors - Anchor: {anchor_factor:.2f}, Positive: {positive_factor:.2f}, Negative: {neg_factor:.2f}")
                            
                            # Compute loss for this triplet
                            loss = contrastive_loss(anchor, positive, negative, margin)
                            total_loss += loss.item()
                            triplet_count += 1
                            
                            # Update prototypes using gradient information (simplified version)
                            if loss.item() > 0.01:  # Only update if loss is significant
                                # Move embeddings to improve their relationships
                                with torch.no_grad():
                                    # Asymmetric update approach with centrality-based weights
                                    direction = positive - anchor
                                    anchor = anchor + anchor_weight * direction
                                    positive = positive - pos_weight * direction
                                    
                                    # Push negative further away
                                    neg_direction = negative - anchor
                                    negative = negative + neg_weight * neg_direction
                                    
                                    # Update the embeddings in the database
                                    self.feature_database[label][i] = anchor.numpy()
                                    self.feature_database[label][j] = positive.numpy()
                                    self.feature_database[neg_label][0] = negative.numpy()
            
            # Update all prototypes
            for label, embeddings in self.feature_database.items():
                if embeddings:
                    self.prototypes[label] = np.mean([e for e in embeddings], axis=0)
            
            # Report loss
            if triplet_count > 0:
                avg_loss = total_loss / triplet_count
                print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f} (across {triplet_count} triplets)")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, No valid triplets found")
        
        # Report improvement 
        print("\nTraining completed. Embedding space changes:")
        for label1, label2 in itertools.combinations(original_prototypes.keys(), 2):
            if label1 in original_prototypes and label2 in original_prototypes:
                # Calculate similarity before training
                before_sim = cosine_similarity(
                    np.array(original_prototypes[label1]).reshape(1, -1),
                    np.array(original_prototypes[label2]).reshape(1, -1)
                )[0][0]
                
                # Calculate similarity after training
                after_sim = cosine_similarity(
                    np.array(self.prototypes[label1]).reshape(1, -1),
                    np.array(self.prototypes[label2]).reshape(1, -1)
                )[0][0]
                
                print(f"'{label1}' vs '{label2}': Before: {before_sim:.4f}, After: {after_sim:.4f}, Delta: {after_sim - before_sim:+.4f}")

    def train_contrastive_optimized(self, epochs: int = 10, margin: float = 1.0) -> None:
        """Train embeddings using optimized contrastive learning for large datasets."""
        # Keep track of original prototypes
        original_prototypes = {label: proto.copy() if isinstance(proto, np.ndarray) else proto.clone() 
                              for label, proto in self.prototypes.items()}
        
        # Calculate initial similarity matrix between all classes
        initial_similarities = {}
        for label1, label2 in itertools.combinations(self.prototypes.keys(), 2):
            sim = cosine_similarity(
                np.array(self.prototypes[label1]).reshape(1, -1),
                np.array(self.prototypes[label2]).reshape(1, -1)
            )[0][0]
            initial_similarities[(label1, label2)] = sim
            print(f"Initial similarity '{label1}' vs '{label2}': {sim:.4f}")
        
        # Identify problematic class pairs (high initial similarity)
        problematic_pairs = {pair: sim for pair, sim in initial_similarities.items() if sim > 0.5}
        if problematic_pairs:
            print("\nDetected problematic class pairs with high similarity:")
            for (label1, label2), sim in problematic_pairs.items():
                print(f"  '{label1}' vs '{label2}': {sim:.4f}")
        
        # Adaptive parameters - stronger negative push for problematic pairs
        base_anchor_weight = 0.01
        base_pos_weight = 0.003
        base_neg_weight = 0.03  # Base negative effect
        problematic_neg_multiplier = 3.0  # Much stronger negative effect for problematic pairs
        
        # Calculate number of possible class pairs (C choose 2)
        num_examples = sum(len(embs) for embs in self.feature_database.values())
        num_classes = len(self.feature_database)
        possible_class_pairs = num_classes * (num_classes - 1) // 2
        
        # Scale triplets based on dataset characteristics
        examples_per_class = num_examples / max(1, num_classes)
        base_triplets = int(math.sqrt(examples_per_class) * possible_class_pairs * 5)
        
        # Ensure reasonable bounds regardless of dataset size
        min_triplets_per_epoch = max(20, possible_class_pairs * 3)  # At least 3 triplets per class pair
        max_triplets_per_epoch = 200  # Cap at reasonable maximum
        
        # Final batch size calculation with extra triplets for problematic pairs
        dynamic_batch_size = min(max_triplets_per_epoch, max(min_triplets_per_epoch, base_triplets))
        if problematic_pairs:
            # Add more triplets to focus on problematic pairs
            dynamic_batch_size = min(max_triplets_per_epoch, dynamic_batch_size + len(problematic_pairs) * 10)
        
        print(f"Training with {epochs} epochs, {dynamic_batch_size} triplets per epoch")
        print(f"Dataset has {num_examples} examples across {num_classes}")
        print(f"Using approximately {dynamic_batch_size // max(1, possible_class_pairs)} triplets per class pair")
        
        # Process all triplets at once for very small datasets
        if num_examples < 30:
            print("Small dataset detected: Using exhaustive triplet exploration")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Sample fresh triplets for each epoch
            triplets = self.sample_triplets(dynamic_batch_size)
            
            # Add extra triplets focusing specifically on problematic pairs
            if problematic_pairs:
                extra_triplets = self.sample_problematic_triplets(problematic_pairs, 
                                                                num_triplets=len(problematic_pairs) * 5)
                if extra_triplets:
                    print(f"Adding {len(extra_triplets)} extra triplets focusing on problematic pairs")
                    triplets.extend(extra_triplets)
            
            if not triplets:
                print("No valid triplets could be generated. Skipping epoch.")
                continue
                
            print(f"Generated {len(triplets)} triplets for training")
            
            # Process triplets
            processed_triplets = 0
            total_loss = 0.0
            
            for triplet in triplets:
                label1, i, anchor, label2, j, positive, neg_label, k, negative, a_factor, p_factor, n_factor = triplet
                processed_triplets += 1
                
                # Detect if this triplet represents a problematic pair
                is_problematic_pair = False
                pair1 = (label1, neg_label) if label1 < neg_label else (neg_label, label1)
                pair2 = (label2, neg_label) if label2 < neg_label else (neg_label, label2)
                
                if pair1 in problematic_pairs or pair2 in problematic_pairs:
                    is_problematic_pair = True
                    # Apply stronger negative push for problematic pairs
                    neg_weight = base_neg_weight * n_factor * problematic_neg_multiplier
                    # Use larger margin for problematic pairs
                    adaptive_margin = margin * 2.0
                    
                    if processed_triplets % 20 == 0:  # Log occasionally
                        print(f"Using extra strong negative push for problematic pair: {pair1 if pair1 in problematic_pairs else pair2}")
                else:
                    # Standard weights for normal pairs
                    neg_weight = base_neg_weight * n_factor
                    adaptive_margin = margin
                
                # ROBUST APPROACH: Adaptive hard negative mining based on loss and epoch
                hard_neg_probability = 0.3 + (0.5 * epoch / epochs)
                if is_problematic_pair:  # Always use hard negative mining for problematic pairs
                    hard_neg_probability = 0.9
                    
                if np.random.random() < hard_neg_probability:
                    hard_negatives = self.mine_hard_negatives(anchor, label1, num_negatives=2)
                    if hard_negatives:
                        # Select hardest negative (or second hardest occasionally for diversity)
                        neg_idx = 0 if len(hard_negatives) == 1 or np.random.random() > 0.3 else 1
                        neg_label, k, negative, _ = hard_negatives[neg_idx]
                        
                        # Print occasionally
                        if processed_triplets % 20 == 0:
                            print(f"Using hard negative mining (prob: {hard_neg_probability:.2f})")
                
                # Calculate weights with less aggressive decay for problematic pairs
                epoch_scale = 1.0 - 0.3 * (epoch / epochs)  # Scale from 1.0 to 0.7
                if is_problematic_pair:
                    epoch_scale = max(0.8, epoch_scale)  # Keep stronger updates for problematic pairs
                    
                anchor_weight = base_anchor_weight * a_factor * epoch_scale
                pos_weight = base_pos_weight * p_factor * epoch_scale
                
                # Compute loss with appropriate margin
                loss = contrastive_loss(anchor, positive, negative, adaptive_margin)
                total_loss += loss.item()
                
                # Update embeddings with correct push directions
                if loss.item() > 0.001 or is_problematic_pair:  # Always update problematic pairs
                    with torch.no_grad():
                        # Store original magnitudes
                        anchor_orig_mag = torch.norm(anchor).item()
                        positive_orig_mag = torch.norm(positive).item()
                        negative_orig_mag = torch.norm(negative).item()
                        
                        # Step 1: Move anchor and positive closer
                        pos_direction = positive - anchor
                        anchor = anchor + anchor_weight * pos_direction
                        positive = positive - pos_weight * pos_direction
                        
                        # Step 2: Push negative AWAY from anchor
                        neg_direction = negative - anchor
                        negative = negative - neg_weight * neg_direction
                        
                        # Step 3: Add explicit orthogonalization (stronger for problematic pairs)
                        anchor_norm = F.normalize(anchor, dim=0)
                        negative_norm = F.normalize(negative, dim=0)
                        
                        # Push negative toward being anti-correlated with anchor
                        orthogonal_strength = 0.1
                        if is_problematic_pair:
                            orthogonal_strength = 0.25  # Stronger orthogonalization
                            
                        orthogonal_direction = -anchor_norm  # Direct opposition
                        negative = negative + orthogonal_strength * orthogonal_direction * negative_orig_mag
                        
                        # Restore magnitudes to prevent embedding collapse
                        anchor = F.normalize(anchor, dim=0) * anchor_orig_mag
                        positive = F.normalize(positive, dim=0) * positive_orig_mag
                        negative = F.normalize(negative, dim=0) * negative_orig_mag
                        
                        # Update the embeddings in the database
                        self.feature_database[label1][i] = anchor.numpy()
                        self.feature_database[label2][j] = positive.numpy()
                        self.feature_database[neg_label][k] = negative.numpy()
            
            # Update all prototypes after each epoch
            for label, embeddings in self.feature_database.items():
                if embeddings:
                    self.prototypes[label] = np.mean([e for e in embeddings], axis=0)
            
            # Report loss
            if processed_triplets > 0:
                avg_loss = total_loss / processed_triplets
                print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f} (across {processed_triplets} triplets)")
        
        # After training completes, apply a final orthogonalization pass for problematic pairs
        if problematic_pairs:
            print("\nApplying final orthogonalization for problematic class pairs...")
            for (label1, label2), sim in problematic_pairs.items():
                # Get updated prototypes
                proto1 = torch.tensor(self.prototypes[label1], dtype=torch.float32)
                proto2 = torch.tensor(self.prototypes[label2], dtype=torch.float32)
                
                # Calculate unit vectors
                proto1_norm = F.normalize(proto1, p=2, dim=0)
                proto2_norm = F.normalize(proto2, p=2, dim=0)
                
                # Push apart with stronger force
                separation_factor = 0.2
                proto1_new = proto1 - separation_factor * proto2_norm
                proto2_new = proto2 - separation_factor * proto1_norm
                
                # Normalize to maintain magnitude
                proto1_mag = torch.norm(proto1).item()
                proto2_mag = torch.norm(proto2).item()
                proto1_new = F.normalize(proto1_new, dim=0) * proto1_mag
                proto2_new = F.normalize(proto2_new, dim=0) * proto2_mag
                
                # Update prototypes
                self.prototypes[label1] = proto1_new.numpy()
                self.prototypes[label2] = proto2_new.numpy()
        
        # Report improvement 
        print("\nTraining completed. Embedding space changes:")
        for label1, label2 in itertools.combinations(original_prototypes.keys(), 2):
            if label1 in original_prototypes and label2 in original_prototypes:
                # Calculate similarity before training
                before_sim = cosine_similarity(
                    np.array(original_prototypes[label1]).reshape(1, -1),
                    np.array(original_prototypes[label2]).reshape(1, -1)
                )[0][0]
                
                # Calculate similarity after training
                after_sim = cosine_similarity(
                    np.array(self.prototypes[label1]).reshape(1, -1),
                    np.array(self.prototypes[label2]).reshape(1, -1)
                )[0][0]
                
                # Highlight problematic pairs in the output
                pair = (label1, label2) if label1 < label2 else (label2, label1)
                highlight = " 👉 PROBLEMATIC PAIR" if pair in problematic_pairs else ""
                
                print(f"'{label1}' vs '{label2}': Before: {before_sim:.4f}, After: {after_sim:.4f}, Delta: {after_sim - before_sim:+.4f}{highlight}")

    def sample_triplets(self, batch_size=20):
        """Dynamically sample triplets for contrastive learning."""
        triplets = []
        
        # Get labels with at least 2 examples
        valid_labels = [label for label, embeddings in self.feature_database.items() 
                       if len(embeddings) >= 2]
        
        if len(valid_labels) < 1:
            return []  # Not enough data for triplets
        
        # Sample batch_size triplets
        for _ in range(batch_size):
            # Sample a random class
            label = np.random.choice(valid_labels)
            embeddings = self.feature_database[label]
            
            # Sample two different indices for anchor and positive
            if len(embeddings) >= 2:
                i, j = np.random.choice(len(embeddings), 2, replace=False)
                anchor = torch.tensor(embeddings[i], dtype=torch.float32)
                positive = torch.tensor(embeddings[j], dtype=torch.float32)
                
                # Sample a negative from a different class
                neg_labels = [l for l in valid_labels if l != label]
                if neg_labels:
                    neg_label = np.random.choice(neg_labels)
                    neg_embeddings = self.feature_database[neg_label]
                    k = np.random.randint(len(neg_embeddings))
                    negative = torch.tensor(neg_embeddings[k], dtype=torch.float32)
                    
                    # Calculate centrality factors as before
                    if label in self.prototypes:
                        class_prototype = torch.tensor(self.prototypes[label], dtype=torch.float32)
                        anchor_norm = F.normalize(anchor, p=2, dim=0, eps=1e-8)
                        positive_norm = F.normalize(positive, p=2, dim=0, eps=1e-8)
                        prototype_norm = F.normalize(class_prototype, p=2, dim=0, eps=1e-8)
                        
                        anchor_centrality = torch.dot(anchor_norm, prototype_norm).item()
                        anchor_centrality = max(-0.999, min(0.999, anchor_centrality))
                        
                        positive_centrality = torch.dot(positive_norm, prototype_norm).item()
                        positive_centrality = max(-0.999, min(0.999, positive_centrality))
                        
                        # Scaling factors
                        anchor_factor = 0.2 * (1.0 - anchor_centrality) + 0.8
                        positive_factor = 0.2 * (1.0 - positive_centrality) + 0.8
                    else:
                        anchor_factor = 1.0
                        positive_factor = 1.0
                    
                    # Add triplet
                    triplets.append((
                        label, i, anchor.clone(), 
                        label, j, positive.clone(), 
                        neg_label, k, negative.clone(),
                        anchor_factor, positive_factor, 1.0
                    ))
        
        return triplets

    def mine_hard_negatives(self, anchor, label, num_negatives=3):
        """Find the hardest negative examples (closest to anchor)."""
        hard_negatives = []
        
        for neg_label, neg_embeddings in self.feature_database.items():
            if neg_label == label or not neg_embeddings:
                continue
                
            # Convert anchor to tensor if needed
            if not isinstance(anchor, torch.Tensor):
                anchor = torch.tensor(anchor, dtype=torch.float32)
            
            # Calculate similarities with all negative examples
            similarities = []
            for idx, neg_emb in enumerate(neg_embeddings):
                neg_tensor = torch.tensor(neg_emb, dtype=torch.float32)
                
                # Safe normalization
                anchor_norm = F.normalize(anchor, p=2, dim=0, eps=1e-8)
                neg_norm = F.normalize(neg_tensor, p=2, dim=0, eps=1e-8)
                
                # Compute similarity
                similarity = torch.dot(anchor_norm, neg_norm).item()
                similarities.append((neg_label, idx, neg_tensor, similarity))
            
            # Sort by similarity (highest first = hardest negatives)
            similarities.sort(key=lambda x: x[3], reverse=True)
            
            # Take the hardest negatives
            hard_negatives.extend(similarities[:min(num_negatives, len(similarities))])
        
        return hard_negatives

    def sample_problematic_triplets(self, problematic_pairs, num_triplets=10):
        """Sample triplets that specifically target problematic class pairs."""
        triplets = []
        
        for _ in range(num_triplets):
            # Randomly select a problematic pair
            if not problematic_pairs:
                return []
                
            pair = random.choice(list(problematic_pairs.keys()))
            label1, label2 = pair
            
            # Get embeddings for both classes
            embeddings1 = self.feature_database[label1]
            embeddings2 = self.feature_database[label2]
            
            if len(embeddings1) < 1 or len(embeddings2) < 1:
                continue
                
            # Select random embeddings from each class
            i = np.random.randint(len(embeddings1))
            k = np.random.randint(len(embeddings2))
            
            anchor = torch.tensor(embeddings1[i], dtype=torch.float32)
            negative = torch.tensor(embeddings2[k], dtype=torch.float32)
            
            # Find a positive example from the same class as anchor
            if len(embeddings1) > 1:
                # Choose a different index for positive
                j_options = [j for j in range(len(embeddings1)) if j != i]
                j = np.random.choice(j_options)
            else:
                # If only one example, use it (not ideal but better than nothing)
                j = i
                
            positive = torch.tensor(embeddings1[j], dtype=torch.float32)
            
            # Add triplet with high weight factors for strong updates
            triplets.append((
                label1, i, anchor.clone(), 
                label1, j, positive.clone(), 
                label2, k, negative.clone(),
                1.0, 1.0, 2.0  # Higher negative factor
            ))
            
        return triplets