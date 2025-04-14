import itertools
import sys
import os
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QMessageBox, QDialog, QProgressBar, QLineEdit, QListWidget,
    QListWidgetItem, QColorDialog, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
import math

from segmenter_scop import Segmenter
from label_predictor import LabelPredictor

from label_dialog import LabelDialog

def load_stylesheet(file_path):
    """Load stylesheet from a file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading stylesheet: {e}")
        return ""

# Subclass QLabel to capture mouse clicks on the image
class ClickableLabel(QLabel):
    clicked = pyqtSignal(object)
    mouse_moved = pyqtSignal(object)  # New signal for mouse move

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.interactions_enabled = False

    def mousePressEvent(self, event):
        if self.interactions_enabled and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self.interactions_enabled):   
            self.mouse_moved.emit(event.pos())
        super().mouseMoveEvent(event)

# Thread to generate the initial proposed point using an existing segmenter
class MaskingThread(QThread):
    result_ready = pyqtSignal(tuple)  # Signal emitting the final result
    error_occurred = pyqtSignal(str)  # Signal for handling errors (optional)

    def __init__(self, segmenter, parent=None):
        super(MaskingThread, self).__init__(parent)
        self.segmenter = segmenter

    def run(self):
        try:
            best_point = self.segmenter.get_best_point()
            self.result_ready.emit(best_point)
        except Exception as e:
            self.error_occurred.emit(str(e))

# Thread to expand the user-selected point into a mask using an existing segmenter
class MaskExpansionThread(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self, segmenter, points, labels):
        super().__init__()
        self.segmenter = segmenter
        self.points = points
        self.labels = labels

    def run(self):
        mask = self.segmenter.propagate_points(self.points, self.labels, update_expanded_mask=True)
        self.result_ready.emit(mask)


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # Image display
        self.image_label = ClickableLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(1000, 700)
        self.image_label.clicked.connect(self.on_image_clicked)
        self.image_label.interactions_enabled = False

        # Buttons
        stylesheet = load_stylesheet("button_styles.qss")
        app.setStyleSheet(stylesheet)

        self.select_button = QPushButton("Select Folder", self)
        self.select_button.clicked.connect(self.select_folder)
        self.select_button.setFixedSize(150, 40)
        self.select_button.setProperty("class", "select-folder-button")
        self.select_button.enterEvent = lambda e: self.on_cursor_over_button()

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_labeling)
        self.start_button.setFixedSize(150, 40)
        self.start_button.setEnabled(False)
        self.start_button.setProperty("class", "start-button")
        self.start_button.enterEvent = lambda e: self.on_cursor_over_button()

        # Navigation buttons
        self.prev_button = QPushButton("<", self)
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setFixedSize(40, 40)
        self.prev_button.setEnabled(False)
        self.prev_button.setProperty("class", "navigation-button")
        self.prev_button.enterEvent = lambda e: self.on_cursor_over_button()

        self.next_button = QPushButton(">", self)
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setFixedSize(40, 40)
        self.next_button.setEnabled(False)
        self.next_button.setProperty("class", "navigation-button")
        self.next_button.enterEvent = lambda e: self.on_cursor_over_button()

        # New buttons for point selection
        self.switch_button = QPushButton("Negative", self)
        self.switch_button.clicked.connect(self.switch_point_type)
        self.switch_button.setFixedSize(150, 40)
        self.switch_button.setEnabled(False)
        self.switch_button.setProperty("class", "switch-button-negative")
        self.switch_button.enterEvent = lambda e: self.on_cursor_over_button()

        self.finish_button = QPushButton("✓", self)
        self.finish_button.clicked.connect(self.on_finish_button_clicked)
        self.finish_button.setFixedSize(40, 40)
        self.finish_button.setEnabled(False)
        self.finish_button.setProperty("class", "finish-button")
        self.finish_button.enterEvent = lambda e: self.on_cursor_over_button()

        # Toggle mask visibility button
        self.toggle_masks_button = QPushButton("👁️", self)
        self.toggle_masks_button.clicked.connect(self.toggle_masks_visibility)
        self.toggle_masks_button.setFixedSize(40, 40)
        self.toggle_masks_button.setEnabled(False)
        self.toggle_masks_button.setProperty("class", "toggle-masks-button")
        self.toggle_masks_button.enterEvent = lambda e: self.on_cursor_over_button()

        # Create a horizontal layout for the bottom buttons
        bottom_button_layout = QHBoxLayout()
        bottom_button_layout.addStretch()
        bottom_button_layout.addWidget(self.start_button)
        bottom_button_layout.addWidget(self.switch_button)
        bottom_button_layout.addWidget(self.finish_button)
        bottom_button_layout.addWidget(self.toggle_masks_button)
        bottom_button_layout.addStretch()

        # Create a container widget for the image and navigation buttons
        image_container = QWidget()
        image_container.setFixedSize(1080, 700)  # Increased width to accommodate buttons
        image_container_layout = QGridLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)
        image_container_layout.setSpacing(0)
        
        # Add image label to the center
        image_container_layout.addWidget(self.image_label, 0, 1)  # Changed to column 1
        
        # Add navigation buttons to corners
        image_container_layout.addWidget(self.prev_button, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        image_container_layout.addWidget(self.next_button, 0, 2, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Main layout: image on left, bottom buttons below
        main_layout = QVBoxLayout()
        
        # Create a horizontal layout for the top row (select folder button)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.select_button)
        top_layout.addStretch()
        
        main_layout.addLayout(top_layout)
        main_layout.addWidget(image_container, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(bottom_button_layout)
        
        self.setLayout(main_layout)

        # Variables
        self.image_list = []
        self.current_index = 0
        self.current_image = None
        self.overlay_image = None
        self.displayed_pixmap = None
        self.segmenter = None
        self.labels = {}
        self.expanded_masks = []
        self.combined_mask_overlay = None

        self.image_label.mouse_moved.connect(self.on_mouse_moved)

        # New variables for multiple points
        self.positive_points = []
        self.negative_points = []
        self.current_point_type = "positive"  # or "negative"
        self.is_selecting_points = False

        # Initially hide the point selection buttons
        self.switch_button.hide()
        self.finish_button.hide()
        self.toggle_masks_button.hide()

        # Initialize additional variables
        self.current_image_path = None
        self.current_label = None
        self.expanded_areas_mask = None
        self.current_mask = None
        self.is_expanding = False
        self.current_mask_area = 0
        self.total_image_area = 0
        self.coverage_ratio = 0
        self.label_predictor = LabelPredictor(confidence_threshold=0.85, min_examples_per_class=5)

        self.current_mode = "creation"  # or "selection"
        self.is_over_mask = False
        self.current_mask_index = None
        self.masks_visible = True

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Images")
        if folder:
            self.image_list = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
            ]
            self.image_list.sort()
            self.current_index = 0
            self.segmenter = None  # Reset segmenter for a new folder

            if self.image_list:
                self.show_image()
                self.start_button.setEnabled(True)
                self.next_button.setEnabled(True)
                self.prev_button.setEnabled(True)

    def start_labeling(self):
        if not self.image_list or self.current_index >= len(self.image_list):
            return
        
        # Enable image interactions and show point selection buttons
        self.image_label.interactions_enabled = True
        self.switch_button.show()
        self.switch_button.setEnabled(False)  # Initially disabled
        self.finish_button.show()
        self.finish_button.setEnabled(False)  # Will be enabled after first positive point
        self.start_button.setEnabled(False)  # Disable Start button until Next Image is pressed
        self.is_selecting_points = True

        # Show the toggle masks button but disable it until first mask is created
        self.toggle_masks_button.show()
        self.toggle_masks_button.setEnabled(False)

        # Create a modal progress dialog
        wait_dialog = QDialog(self)
        wait_dialog.setWindowTitle("Generating Proposed Point")
        wait_dialog.setModal(True)
        wait_layout = QVBoxLayout()
        wait_label = QLabel("Please wait while the proposed point is being generated...")
        wait_layout.addWidget(wait_label)
        progress_bar = QProgressBar(wait_dialog)
        progress_bar.setRange(0, 0)  # Indeterminate mode
        wait_layout.addWidget(progress_bar)
        wait_dialog.setLayout(wait_layout)
        wait_dialog.resize(400, 150)
        wait_dialog.show()
        QApplication.processEvents()

        QTimer.singleShot(100, lambda: self.initialize_segmenter_and_start_thread(wait_dialog))
        wait_dialog.exec()

    def initialize_segmenter_and_start_thread(self, wait_dialog):
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        sam_checkpoint = "checkpoints/vit_b_coralscop.pth"
        sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        current_image_path = self.image_list[self.current_index]

        image = cv2.imread(current_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image

        if self.segmenter is None:
            self.segmenter = Segmenter(self.current_image, sam_checkpoint, sam2_checkpoint, sam2_cfg)

        self.masking_thread = MaskingThread(self.segmenter)
        self.masking_thread.result_ready.connect(lambda best_point: self.on_masking_complete(best_point, wait_dialog))
        self.masking_thread.start()

    def on_masking_complete(self, best_point, wait_dialog):
        wait_dialog.accept()
        if (best_point):
            print(f"Proposed point: {best_point}")
            self.suggested_point = best_point
        else:
            print("No valid point proposed.")
            self.suggested_point = None
        self.show_image(overlay_point=self.suggested_point)

    def on_image_clicked(self, pos):
        if self.current_image is None or self.displayed_pixmap is None:
            return

        # Convert click position to image coordinates
        current_point = self.get_image_coordinates(pos)
        if current_point is None:
            return
        
        # Check which mode we're in
        if hasattr(self, 'current_mode'):
            # Don't do anything on clicks in visualization mode
            if self.current_mode == "visualization":
                return
            elif self.current_mode == "selection":
                mask_index = self.get_mask_at_position(current_point)
                if mask_index is not None:
                    # Show context menu for the selected mask
                    self.current_mask_index = mask_index
                    self.show_mask_context_menu(pos, mask_index)
                    return
            elif self.current_mode == "creation":
                # We're in creation mode - first check if we're over a mask
                mask_index = self.get_mask_at_position(current_point)
                if mask_index is not None and self.is_over_mask:
                    # Show context menu for the selected mask
                    self.current_mask_index = mask_index
                    self.show_mask_context_menu(pos, mask_index)
                else:
                    # Normal point placement behavior
                    if self.current_point_type == "positive":
                        self.positive_points.append(current_point)
                        print(f"Positive point added at: {current_point}")
                        self.finish_button.setEnabled(True)
                        self.switch_button.setEnabled(True)
                    else:
                        self.negative_points.append(current_point)
                        print(f"Negative point added at: {current_point}")
                    
                    # Update preview with the new point
                    self.update_preview_with_points()
        else:
            # Fallback if mode is not set
            if self.current_point_type == "positive":
                self.positive_points.append(current_point)
                print(f"Positive point added at: {current_point}")
                self.finish_button.setEnabled(True)
                self.switch_button.setEnabled(True)
            else:
                self.negative_points.append(current_point)
                print(f"Negative point added at: {current_point}")
            
            # Update preview with the new point
            self.update_preview_with_points()

    def update_preview_with_points(self):
        """Update display with current points and preview mask"""
        # Start with original image and add cached overlay if masks are visible
        overlay_image = self.current_image.copy()
        if self.masks_visible and self.combined_mask_overlay is not None:
            overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.6, 0)

        # Draw all current points
        for point in self.positive_points:
            cv2.circle(overlay_image, point, 4, (0, 255, 0), -1)  # Green for positive
        for point in self.negative_points:
            cv2.circle(overlay_image, point, 4, (255, 0, 0), -1)  # Red for negative

        # Show dynamic expansion with current points
        if self.positive_points or self.negative_points:
            points = np.array(self.positive_points + self.negative_points)
            labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points))
            preview_mask = self.segmenter.propagate_points(points, labels, update_expanded_mask=False)
            
            if preview_mask is not None:
                colored_preview = np.zeros_like(overlay_image)
                colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray for preview
                overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)

        # Add suggested point if it exists
        if hasattr(self, 'suggested_point') and self.suggested_point:
            row, col = self.suggested_point
            cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 0, 0), 2)
            cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 0, 0), 2)

        self.update_display(overlay_image)

    def delete_mask(self, mask_index):
        """Delete a mask and update the display"""
        if not (0 <= mask_index < len(self.expanded_masks)):
            return
            
        mask, label, _ = self.expanded_masks[mask_index]
        
        # Remove from expanded masks
        self.expanded_masks.pop(mask_index)
        
        # Regenerate the combined mask overlay
        self.regenerate_combined_mask_overlay()
        
        # Reset selection state
        self.current_mask_index = None
        self.current_mode = "creation"
        self.is_over_mask = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Print informative message
        print(f"Deleted mask with label '{label}'")
        
        # Update the display
        self.update_display_with_current_state()

    def change_mask_label(self, mask_index):
        """Change the label of a mask"""
        if not (0 <= mask_index < len(self.expanded_masks)):
            return
            
        mask, old_label, _ = self.expanded_masks[mask_index]
        
        # Show label dialog to pick a new label
        dialog = LabelDialog(self.labels, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_label = dialog.selected_label
            if new_label is None:
                new_label = dialog.new_label_edit.text()
                if new_label and dialog.chosen_color:
                    self.labels[new_label] = dialog.chosen_color
            
            if new_label and new_label in self.labels:
                color = self.labels[new_label]
                
                # Update the mask in the list
                self.expanded_masks[mask_index] = (mask, new_label, color)
                
                # Update the combined overlay
                self.regenerate_combined_mask_overlay()
                
                print(f"Changed mask label from '{old_label}' to '{new_label}'")
                
                # Update the display
                self.update_display_with_current_state()

    def regenerate_combined_mask_overlay(self):
        """Regenerate the combined mask overlay from all masks"""
        if not self.expanded_masks:
            self.combined_mask_overlay = None
            return
    
        self.combined_mask_overlay = np.zeros_like(self.current_image)
        
        for mask, _, color in self.expanded_masks:
            colored_mask = np.zeros_like(self.current_image)
            colored_mask[mask > 0] = [color.red(), color.green(), color.blue()]
            self.combined_mask_overlay[mask > 0] = colored_mask[mask > 0]

    def show_mask_context_menu(self, pos, mask_index):
        """Show context menu for the selected mask"""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction  # Import QAction from QtGui instead of QtWidgets
        
        if mask_index is None or mask_index >= len(self.expanded_masks):
            return
        
        mask, label, color = self.expanded_masks[mask_index]
        
        # Create context menu
        context_menu = QMenu(self)
        
        # Add actions
        delete_action = QAction(f"Delete '{label}' mask", self)
        change_label_action = QAction(f"Change label (current: '{label}')", self)
        
        # Add actions to menu
        context_menu.addAction(delete_action)
        context_menu.addAction(change_label_action)
        
        # Connect actions to handlers
        delete_action.triggered.connect(lambda: self.delete_mask(mask_index))
        change_label_action.triggered.connect(lambda: self.change_mask_label(mask_index))
        
        # Show context menu at cursor position
        context_menu.exec(self.mapToGlobal(pos))

    def on_mask_expansion_complete(self, mask, stored_positive_points):
        if mask is None:
            print("No mask generated.")
            self.is_expanding = False  # Reset expansion state
            self.finish_button.setEnabled(True)  # Re-enable the finish button
            return

        # Print detailed database status at the beginning
        self.label_predictor.print_database_status()
            
        # Get predictions for current mask BEFORE showing dialog (to show initial predictions)
        if hasattr(self, 'label_predictor'):
            print("\n===== INITIAL MASK PREDICTION =====")
            
            if not self.labels:
                print("No labels defined yet. This will be the first mask.")
            elif not hasattr(self.label_predictor, 'feature_database') or len(self.label_predictor.feature_database) == 0:
                print("Feature database is empty. No predictions possible.")
            else:
                # Get predictions for current mask (even if we don't have enough examples)
                try:
                    # Try to get raw similarities for all classes
                    features = self.label_predictor._extract_features(self.current_image, mask)
                    
                    print(f"Available labels: {list(self.labels.keys())}")
                    print(f"Current min examples threshold: {self.label_predictor.min_examples_per_class}")
                    
                    # Get predictions with all available classes
                    all_predictions = []
                    
                    # Try to compute similarities even if below min_examples threshold
                    if hasattr(self.label_predictor, 'prototypes') and self.label_predictor.prototypes:
                        for label, prototype in self.label_predictor.prototypes.items():
                            current_count = len(self.label_predictor.feature_database.get(label, []))
                            try:
                                # Extract spatial features from the image
                                spatial_features = self.label_predictor._extract_features(self.current_image, mask)
                                
                                # Calculate mask embedding (average of features inside the mask)
                                mask_embedding = self.label_predictor._compute_mask_embedding(spatial_features, mask)
                                
                                if mask_embedding is None:
                                    print(f"Could not calculate embedding for '{label}': Empty mask")
                                    continue
                                    
                                # Convert to torch tensors with correct dimensions
                                feat_tensor = torch.tensor(mask_embedding, dtype=torch.float32)
                                proto_tensor = torch.tensor(prototype, dtype=torch.float32)

                                # Check for problematic values before normalization
                                if torch.isnan(feat_tensor).any() or torch.isinf(feat_tensor).any():
                                    print(f"Warning: NaN or Inf in feature tensor before normalization")
                                    # Replace problematic values with zeros
                                    feat_tensor = torch.nan_to_num(feat_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

                                if torch.isnan(proto_tensor).any() or torch.isinf(proto_tensor).any():
                                    print(f"Warning: NaN or Inf in prototype tensor before normalization")
                                    # Replace problematic values with zeros
                                    proto_tensor = torch.nan_to_num(proto_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

                                # Then normalize with epsilon for numerical stability
                                feat_norm = F.normalize(feat_tensor, dim=0, eps=1e-8)
                                proto_norm = F.normalize(proto_tensor, dim=0, eps=1e-8)

                                # Safe dot product with error handling
                                try:
                                    similarity = torch.dot(feat_norm, proto_norm).item()
                                    if math.isnan(similarity) or math.isinf(similarity):
                                        print(f"Warning: Similarity calculation produced NaN or Inf, defaulting to 0.0")
                                        similarity = 0.0
                                except Exception as e:
                                    print(f"Error in similarity calculation: {e}")
                                    similarity = 0.0

                                all_predictions.append((label, similarity, current_count))
                            except Exception as e:
                                print(f"Could not calculate similarity for '{label}': {e}")
                                
                        # Sort by similarity
                        all_predictions.sort(key=lambda x: x[1], reverse=True)
                        
                        if all_predictions:
                            print("Raw similarity scores (before applying threshold):")
                            for label, similarity, current_count in all_predictions:
                                sufficient = "✓" if current_count >= self.label_predictor.min_examples_per_class else "✗"
                                print(f"  - '{label}': {similarity:.4f} ({similarity*100:.1f}%) [Examples: {current_count} {sufficient}]")
                    
                    # Now get the official predictions that pass the threshold
                    predictions = self.label_predictor.predict_label(self.current_image, mask)
                    if predictions:
                        print("\nOfficial predictions (after applying threshold):")
                        for label, prob in predictions:
                            print(f"  - '{label}': {prob:.4f} ({prob*100:.1f}%)")
                        
                        # Check if auto-assignment would happen
                        should_auto, auto_label = self.label_predictor.should_auto_assign(self.current_image, mask)
                        print(f"\nWould auto-assign: {should_auto}, Label: '{auto_label}'")
                        print(f"Confidence threshold: {self.label_predictor.confidence_threshold}")
                    else:
                        print("No official predictions available - insufficient examples per class")
                    
                except Exception as e:
                    print(f"Error getting predictions: {e}")
                
            print("==================================\n")

        dialog = LabelDialog(self.labels, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            label = dialog.selected_label
            if label is None:
                label = dialog.new_label_edit.text()
                if label and dialog.chosen_color:
                    self.labels[label] = dialog.chosen_color
            color = self.labels.get(label)
            if color:
                # Update current_mask and current_label for future reference
                self.current_mask = mask
                self.current_label = label
                
                # Keep storing in expanded_masks for future reference
                self.expanded_masks.append((mask, label, color))

                # Enable the toggle masks button after first mask is created
                if not self.toggle_masks_button.isEnabled():
                    self.toggle_masks_button.setEnabled(True)
                
                # Create overlay image BEFORE using it
                overlay_image = self.current_image.copy()
                
                # Update the combined overlay for efficient display
                if self.combined_mask_overlay is None:
                    self.combined_mask_overlay = np.zeros_like(self.current_image)
                
                colored_mask = np.zeros_like(overlay_image)
                colored_mask[mask > 0] = [color.red(), color.green(), color.blue()]
                self.combined_mask_overlay[mask > 0] = colored_mask[mask > 0]
                
                # Update display with the new combined overlay and suggested point
                overlay_image = cv2.addWeighted(self.current_image, 1.0, self.combined_mask_overlay, 0.6, 0)
                
                # Update point selector with the selected points and mask
                if hasattr(self, 'segmenter') and hasattr(self.segmenter, 'point_selector'):
                    # Find the mask ID that contains our first positive point
                    for mask_obj in self.segmenter.masks:
                        if mask_obj['segmentation'][stored_positive_points[0][0], stored_positive_points[0][1]]:
                            self.segmenter.point_selector.update_selection_state(
                                stored_positive_points[0], 
                                id(mask_obj)
                            )
                            break
                
                # Get new candidate points with updated state
                if hasattr(self, 'segmenter'):
                    self.suggested_point = self.segmenter.get_best_point()
                
                self.update_display(overlay_image)
                
                # Calculate and display coverage statistics
                self.current_mask_area = np.sum(mask)
                self.total_image_area = self.current_image.shape[0] * self.current_image.shape[1]
                self.coverage_ratio = self.current_mask_area / self.total_image_area
                
                # DEBUG: Print feature extraction progress
                print(f"\nAdding example for label '{label}'...")
                try:
                    # Train predictor with the new example
                    self.label_predictor.add_example(self.current_image, mask, label)
                    print(f"✓ Successfully added example for label: {label}")
                    
                    # Show detailed database status after adding
                    self.label_predictor.print_database_status(detailed=True)
                    
                    # Train embeddings using contrastive learning if we have enough examples
                    class_counts = {label: len(examples) for label, examples in 
                                   self.label_predictor.feature_database.items()}
                    
                    # Check if we have enough data to perform contrastive training
                    # We need at least 2 classes with examples
                    classes_with_examples = sum(1 for count in class_counts.values() if count > 0)
                    
                    if classes_with_examples >= 2:
                        print("\n===== TRAINING EMBEDDINGS WITH CONTRASTIVE LEARNING =====")
                        print(f"Classes with examples: {classes_with_examples}")
                        
                        # List all classes and their example counts
                        print("Label Distribution:")
                        for label, count in class_counts.items():
                            print(f"  - '{label}': {count} examples")
                        
                        # Use more epochs for significant updates
                        self.label_predictor.train_contrastive_optimized(epochs=20, margin=0.5)
                        
                        # Show similarities between classes after training
                        print("\n===== POST-TRAINING SIMILARITY ANALYSIS =====")
                        for label1, label2 in itertools.combinations(self.label_predictor.prototypes.keys(), 2):
                            if label1 in self.label_predictor.prototypes and label2 in self.label_predictor.prototypes:
                                proto1 = self.label_predictor.prototypes[label1]
                                proto2 = self.label_predictor.prototypes[label2]
                                
                                # Calculate similarity between prototypes
                                similarity = cosine_similarity(
                                    np.array(proto1).reshape(1, -1),
                                    np.array(proto2).reshape(1, -1)
                                )[0][0]
                                
                                print(f"Similarity between '{label1}' and '{label2}': {similarity:.4f}")
                        
                        print("===== CONTRASTIVE TRAINING COMPLETE =====\n")
                    else:
                        print("\nNot enough classes for contrastive learning yet.")
                        print(f"Need at least 2 classes with examples (current: {classes_with_examples})")
                    
                    # Show updated predictions
                    print("\n===== UPDATED PREDICTIONS =====")
                    predictions = self.label_predictor.predict_label(self.current_image, mask)
                    
                    if predictions:
                        for label, prob in predictions:
                            print(f"  - '{label}': {prob:.4f} ({prob*100:.1f}%)")
                    else:
                        print("No predictions available - need more examples")
                    print("===============================\n")
                    
                except Exception as e:
                    print(f"✗ Failed to add example or train contrastive model: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # ...rest of existing code...
        # Reset expansion state regardless of dialog result
        self.is_expanding = False
        
        # Re-enable the finish button for the next mask
        self.finish_button.setEnabled(True)
        
        # Instead of automatically starting a new labeling cycle, just prepare for the next one
        # This way the user can place points for the next mask at their own pace
        self.switch_button.setEnabled(False)
        if self.current_point_type == "negative":
            self.switch_point_type()  # Reset to positive if it was negative

    def toggle_masks_visibility(self):
        """Toggle the visibility of all masks"""
        self.masks_visible = not self.masks_visible
        
        # Update button appearance and mode
        if self.masks_visible:
            self.toggle_masks_button.setText("👁️❌")
            # Return to creation mode when showing masks
            self.current_mode = "creation" 
        else:
            # When hiding masks, switch to visualization mode
            self.toggle_masks_button.setText("👁️")
            self.current_mode = "visualization"
            self.is_over_mask = False
            self.current_mask_index = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Update the display
        if self.current_image is not None:
            self.update_display_with_current_state()

    def get_mask_at_position(self, pos):
        """Determine which mask (if any) is at the given position"""
        if not self.expanded_masks or not self.masks_visible:
            return None
        
        x, y = pos
        
        # Check each mask in reverse order (newest first)
        for i in range(len(self.expanded_masks) - 1, -1, -1):
            mask, _, _ = self.expanded_masks[i]
            
            # Check if the position is within this mask
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                return i
        
        return None

    def update_display_with_current_state(self):
        """Update the display with current masks and points"""
        # Start with the original image
        overlay_image = self.current_image.copy()
        
        # Add all expanded masks if visible
        if self.masks_visible and self.expanded_masks:
            # Generate a fresh overlay
            combined_overlay = np.zeros_like(self.current_image)
            for i, (mask, _, color) in enumerate(self.expanded_masks):
                # Highlight selected mask with brighter color
                alpha = 0.8 if i == self.current_mask_index else 0.6
                colored_mask = np.zeros_like(overlay_image)
                colored_mask[mask > 0] = [color.red(), color.green(), color.blue()]
                combined_overlay[mask > 0] = colored_mask[mask > 0]
            
            overlay_image = cv2.addWeighted(overlay_image, 1.0, combined_overlay, 0.6, 0)
        
        # Draw points in creation mode
        if self.current_mode == "creation":
            # Draw all current points
            for point in self.positive_points:
                cv2.circle(overlay_image, point, 4, (0, 255, 0), -1)  # Green for positive
            for point in self.negative_points:
                cv2.circle(overlay_image, point, 4, (255, 0, 0), -1)  # Red for negative
            
            # Add suggested point if it exists
            if hasattr(self, 'suggested_point') and self.suggested_point:
                row, col = self.suggested_point
                cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 0, 0), 2)
                cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 0, 0), 2)
        
        # Update the display
        self.update_display(overlay_image)

    def on_mouse_moved(self, pos):
        """Handle mouse movement - determine mode and update preview"""
        # Skip if not in interactive mode
        if not self.is_selecting_points or self.current_image is None:
            return
        
        # Get image coordinates from screen coordinates
        image_pos = self.get_image_coordinates(pos)
        
        # Special handling for cursor outside the image area
        if image_pos is None:
            # If we have points, show expansion with just the existing points
            if self.positive_points or self.negative_points:
                # Start with original image and add cached overlay if visible
                overlay_image = self.current_image.copy()
                if self.masks_visible and self.combined_mask_overlay is not None:
                    overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.6, 0)
                
                # Create points array with only existing points
                points = np.array(self.positive_points + self.negative_points)
                labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points))
                
                # Generate preview mask using only existing points
                preview_mask = self.segmenter.propagate_points(points, labels, update_expanded_mask=False)
                
                # Add the preview mask
                if preview_mask is not None:
                    colored_preview = np.zeros_like(overlay_image)
                    colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray for preview
                    overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)
                
                # Draw all current points
                for point in self.positive_points:
                    cv2.circle(overlay_image, point, 4, (0, 255, 0), -1)
                for point in self.negative_points:
                    cv2.circle(overlay_image, point, 4, (255, 0, 0), -1)
                
                # Add suggested point if it exists
                if hasattr(self, 'suggested_point') and self.suggested_point and self.current_mode == "creation":
                    row, col = self.suggested_point
                    cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 0, 0), 2)
                    cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 0, 0), 2)
                    
                # Update the display
                self.update_display(overlay_image)
            return
        
        # If we reach here, the cursor is inside the image
        # Check which mode we're in
        if self.current_mode == "creation":
            # Call the dynamic expansion function with the current position
            self.dynamic_expand(pos)
        elif self.current_mode == "selection":
            # Handle hover over masks
            mask_index = self.get_mask_at_position(image_pos)
            if mask_index is not None:
                if not self.is_over_mask or self.current_mask_index != mask_index:
                    self.is_over_mask = True
                    self.current_mask_index = mask_index
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                    # Update display to highlight the mask
                    self.update_display_with_current_state()
            else:
                if self.is_over_mask:
                    self.is_over_mask = False
                    self.current_mask_index = None
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                    # Update display to remove highlight
                    self.update_display_with_current_state()

    def on_cursor_over_button(self):
        """Called when cursor enters any button"""
        if self.is_selecting_points:
            # If in visualization mode (masks hidden), don't show any masks
            if self.current_mode == "visualization" or not self.masks_visible:
                # Just show the base image without any masks
                overlay_image = self.current_image.copy()
                
                # Add suggested point if needed in this mode
                if hasattr(self, 'suggested_point') and self.suggested_point and self.current_mode == "creation":
                    row, col = self.suggested_point
                    cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 0, 0), 2)
                    cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 0, 0), 2)
                    
                self.update_display(overlay_image)
                return
            
            # For regular mode with visible masks - continue with existing behavior
            if self.positive_points or self.negative_points:
                points = np.array(self.positive_points + self.negative_points)
                labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points))
                preview_mask = self.segmenter.propagate_points(points, labels, update_expanded_mask=False)
                
                # Start with original image and add cached overlay if it exists
                overlay_image = self.current_image.copy()
                if self.combined_mask_overlay is not None:
                    overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.6, 0)
                
                # Add the preview mask if it exists
                if preview_mask is not None:
                    colored_preview = np.zeros_like(overlay_image)
                    colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray instead of green
                    overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)
                
                # Draw all current points
                for point in self.positive_points:
                    cv2.circle(overlay_image, point, 4, (0, 255, 0), -1)  # Green for positive
                for point in self.negative_points:
                    cv2.circle(overlay_image, point, 4, (255, 0, 0), -1)  # Red for negative
                
                # Add suggested point if it exists
                if hasattr(self, 'suggested_point') and self.suggested_point:
                    row, col = self.suggested_point
                    cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 0, 0), 2)
                    cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 0, 0), 2)
                
                self.update_display(overlay_image)
            else:
                # If no points are placed, just show the base image with any existing overlay
                overlay_image = self.current_image.copy()
                if self.combined_mask_overlay is not None and self.masks_visible:
                    overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.6, 0)
                
                # Add suggested point if it exists
                if hasattr(self, 'suggested_point') and self.suggested_point:
                    row, col = self.suggested_point
                    cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 0, 0), 2)
                    cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 0, 0), 2)
                
                self.update_display(overlay_image)

    def dynamic_expand(self, pos):
        """Handle dynamic expansion with cursor position"""
        # Skip if not in creation mode or not selecting points
        if not self.is_selecting_points or self.current_image is None or self.displayed_pixmap is None:
            return

        # Get image coordinates for the cursor position
        cursor_point = self.get_image_coordinates(pos)
        if cursor_point is None:
            return
            
        # First check if we're over an existing mask
        mask_index = self.get_mask_at_position(cursor_point)
        
        # If over a mask, switch to selection behavior
        if mask_index is not None and self.masks_visible:
            # If we just entered a mask, switch to pointing hand cursor
            if not self.is_over_mask or self.current_mask_index != mask_index:
                self.is_over_mask = True
                self.current_mask_index = mask_index
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                # Update display to highlight the mask
                self.update_display_with_current_state()
            # Skip dynamic expansion while over a mask
            return
        else:
            # If we just left a mask, reset cursor and selection state
            if self.is_over_mask:
                self.is_over_mask = False
                self.current_mask_index = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                # Update display to remove highlight
                self.update_display_with_current_state()
        
        # Start with original image and add cached overlay if it exists
        overlay_image = self.current_image.copy()
        if self.combined_mask_overlay is not None:
            overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.6, 0)

        # Draw all current points
        for point in self.positive_points:
            cv2.circle(overlay_image, point, 4, (0, 255, 0), -1)  # Green for positive
        for point in self.negative_points:
            cv2.circle(overlay_image, point, 4, (255, 0, 0), -1)  # Red for negative

        # Create points array with all current points plus cursor
        points = np.array(self.positive_points + self.negative_points + [cursor_point])
        cursor_label = 0 if self.current_point_type == "negative" else 1
        labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points) + [cursor_label])
        preview_mask = self.segmenter.propagate_points(points, labels, update_expanded_mask=False)

        # Add the preview mask if it exists
        if preview_mask is not None:
            colored_preview = np.zeros_like(overlay_image)
            colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray instead of green
            overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)

        # Draw cursor point
        cursor_color = (0, 255, 0) if self.current_point_type == "positive" else (255, 0, 0)
        cv2.circle(overlay_image, cursor_point, 4, cursor_color, -1)

        # Add suggested point if it exists
        if hasattr(self, 'suggested_point') and self.suggested_point:
            row, col = self.suggested_point
            cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 0, 0), 2)
            cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 0, 0), 2)

        self.update_display(overlay_image)

    def dynamic_expand_with_negative(self, pos):
        if self.current_image is None or self.displayed_pixmap is None:
            return

        cursor_point = self.get_image_coordinates(pos)
        if cursor_point is None:
            return

        # Two point expansion: fixed positive and moving negative
        points = np.array([self.positive_points[0], cursor_point])
        labels = np.array([1, 0])
        
        preview_mask = self.segmenter.propagate_points(points, labels, update_expanded_mask=False)
        
        # Start with original image and add cached overlay if it exists
        overlay_image = self.current_image.copy()
        if self.combined_mask_overlay is not None:
            overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.6, 0)
        
        # Add the preview mask
        if preview_mask is not None:
            colored_preview = np.zeros_like(overlay_image)
            colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray instead of green
            overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)
        
        # Draw the points - moved after all overlays
        pos_x, pos_y = self.positive_points[0]
        cv2.circle(overlay_image, (pos_x, pos_y), 4, (0, 255, 0), -1)  # Green for positive
        cv2.circle(overlay_image, cursor_point, 4, (255, 0, 0), -1)  # Red for negative

        # Add suggested point if it exists (in case it's needed in this mode)
        if hasattr(self, 'suggested_point') and self.suggested_point:
            row, col = self.suggested_point
            cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 0, 0), 2)
            cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 0, 0), 2)

        self.update_display(overlay_image)

    def update_display(self, overlay_image):
        """Helper method to update the display with a new image"""
        height, width, channel = overlay_image.shape
        bytes_per_line = 3 * width
        qimage = QImage(overlay_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.displayed_pixmap = scaled_pixmap
        self.image_label.setPixmap(scaled_pixmap)

    def get_image_coordinates(self, pos):
        """Helper function to convert screen coordinates to image coordinates"""
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap_width = self.displayed_pixmap.width()
        pixmap_height = self.displayed_pixmap.height()

        offset_x = (label_width - pixmap_width) / 2
        offset_y = (label_height - pixmap_height) / 2

        if not (offset_x <= pos.x() <= offset_x + pixmap_width and 
                offset_y <= pos.y() <= offset_y + pixmap_height):
            return None

        original_h, original_w, _ = self.current_image.shape
        ratio_x = original_w / pixmap_width
        ratio_y = original_h / pixmap_height

        orig_x = int((pos.x() - offset_x) * ratio_x)
        orig_y = int((pos.y() - offset_y) * ratio_y)
        
        return (orig_x, orig_y)

    def show_image(self, overlay_point=None):
        if not self.image_list or self.current_index >= len(self.image_list):
            print("No image available to display.")
            return

        # Start with the original image
        if self.current_image is None:
            current_image_path = self.image_list[self.current_index]
            image = cv2.imread(current_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_image = image

        # Create a fresh overlay with the original image
        image = self.current_image.copy()

        # Add all expanded masks
        if self.expanded_masks:
            for mask, label, color in self.expanded_masks:
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0] = [color.red(), color.green(), color.blue()]
                image = cv2.addWeighted(image, 1.0, colored_mask, 0.6, 0)

        # Store the current state of the overlay
        self.overlay_image = image.copy()

        # Add the suggested point if it exists
        if overlay_point:
            row, col = overlay_point
            line_length = 6
            line_color = (255, 0, 0)
            line_thickness = 2
            cv2.line(image, (col, row - line_length), (col, row + line_length), line_color, line_thickness)
            cv2.line(image, (col - line_length, row), (col + line_length, row), line_color, line_thickness)

        # Update the display
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.displayed_pixmap = scaled_pixmap
        self.image_label.setPixmap(scaled_pixmap)

    def next_image(self):
        if not self.image_list or self.current_index >= len(self.image_list):
            QMessageBox.information(self, "Finished", "You have finished labeling all the images.")
            self.reset_viewer()
            return

        # If we have started labeling (segmenter exists), ask for confirmation
        if self.segmenter is not None:
            confirmation = QMessageBox(self)
            confirmation.setWindowTitle("Next Image")
            confirmation.setText("Do you want to save the current labeling and move to the next image?")
            confirmation.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            response = confirmation.exec()
            
            if response == QMessageBox.StandardButton.Cancel:
                return
            elif response == QMessageBox.StandardButton.Yes:
                self.save_image(self.image_list[self.current_index])

        # Disable image interactions until Start button is pressed again
        self.image_label.interactions_enabled = False
        
        # Reset all the labeling-related variables
        self.segmenter = None
        self.expanded_masks = []
        self.overlay_image = None
        self.suggested_point = None
        self.combined_mask_overlay = None  # Clear the combined mask overlay when moving to next image
        
        # Hide the point selection buttons and enable Start button
        self.switch_button.hide()
        self.finish_button.hide()
        self.toggle_masks_button.hide()
        self.toggle_masks_button.setEnabled(False)
        self.start_button.setEnabled(True)
        
        # Move to next image
        self.current_index += 1
        
        if self.current_index < len(self.image_list):
            # Load and display the new image
            current_image_path = self.image_list[self.current_index]
            image = cv2.imread(current_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_image = image
            self.show_image()
        else:
            QMessageBox.information(self, "Finished", "You have finished labeling all the images.")
            self.reset_viewer()

    def save_image(self, image_path):
        if not self.expanded_masks:  # If no masks were created, don't save anything
            return
            
        # Create a black background image
        height, width, _ = self.current_image.shape
        mask_image = np.zeros((height, width, 3), dtype=np.uint8)  # Black background
        
        # Add all expanded masks with their respective colors
        for mask, label, color in self.expanded_masks:
            colored_mask = np.zeros_like(mask_image)
            colored_mask[mask > 0] = [color.red(), color.green(), color.blue()]
            # Use direct assignment instead of addWeighted since we want solid colors
            mask_image[mask > 0] = colored_mask[mask > 0]
        
        # Save the mask image
        save_folder = os.path.join(os.getcwd(), "labeled_masks")
        os.makedirs(save_folder, exist_ok=True)
        image_name = os.path.basename(image_path)
        base_name, ext = os.path.splitext(image_name)
        save_path = os.path.join(save_folder, f"{base_name}_mask{ext}")
        
        # Convert from RGB to BGR for cv2.imwrite
        mask_image_bgr = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, mask_image_bgr)

    def reset_viewer(self):
        self.image_label.clear()
        self.image_list = []
        self.current_index = 0
        self.start_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.prev_button.setEnabled(False)

    def closeEvent(self, event):
        if self.image_list and 0 <= self.current_index < len(self.image_list):
            confirmation = QMessageBox(self)
            confirmation.setWindowTitle("Save Image")
            confirmation.setText("Do you want to save the current image before exiting?")
            confirmation.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            confirmation.setDefaultButton(QMessageBox.StandardButton.No)

            user_response = confirmation.exec()
            if user_response == QMessageBox.StandardButton.Yes:
                self.save_image(self.image_list[self.current_index])
                event.accept()
            elif user_response == QMessageBox.StandardButton.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def switch_point_type(self):
        """Switch between positive and negative point selection"""
        if self.current_point_type == "positive":
            self.current_point_type = "negative"
            self.switch_button.setText("Positive") 
            self.switch_button.setProperty("class", "switch-button-positive")
        else:
            self.current_point_type = "positive"
            self.switch_button.setText("Negative")
            self.switch_button.setProperty("class", "switch-button-negative")
        
        # Force style refresh
        self.switch_button.style().unpolish(self.switch_button)
        self.switch_button.style().polish(self.switch_button)

    def on_finish_button_clicked(self):
        """Handle finish button click - expand mask and update UI"""
        if not self.is_expanding and self.positive_points:
            # Stop any ongoing expansion
            if hasattr(self, 'expansion_thread') and self.expansion_thread and self.expansion_thread.isRunning():
                self.expansion_thread.terminate()
                self.expansion_thread.wait()
            
            # Combine positive and negative points
            points = np.array(self.positive_points + self.negative_points)
            # Create appropriate labels (1 for positive points, 0 for negative points)
            labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points))
            
            # Store positive points for reference
            stored_positive_points = self.positive_points.copy()
            
            # Create expansion thread with correct points and labels format
            self.expansion_thread = MaskExpansionThread(
                self.segmenter,
                points,
                labels
            )
            
            # Connect to result_ready signal instead of finished signal
            self.expansion_thread.result_ready.connect(
                lambda mask: self.on_mask_expansion_complete(mask, stored_positive_points)
            )
            
            # Start expansion
            self.is_expanding = True
            self.expansion_thread.start()
            
            # Update UI
            self.finish_button.setEnabled(False)
            
            # REMOVED: Don't add examples here as they're added in on_mask_expansion_complete
            # This was causing duplicate examples in the database
            
            # Clear points for next mask
            self.positive_points = []
            self.negative_points = []

    def on_expansion_finished(self, mask):
        """Handle mask expansion completion"""
        if mask is not None:
            # Update current mask
            self.current_mask = mask
            
            # Update expanded areas mask
            if not hasattr(self, 'expanded_areas_mask') or self.expanded_areas_mask is None:
                self.expanded_areas_mask = mask.copy()
            else:
                self.expanded_areas_mask = np.logical_or(self.expanded_areas_mask, mask)
            
            # Calculate coverage stats
            self.current_mask_area = np.sum(mask)
            self.total_image_area = self.current_image.shape[0] * self.current_image.shape[1]
            self.coverage_ratio = self.current_mask_area / self.total_image_area
            
            # Output coverage info to console
            print(f"Current Mask Coverage: {self.coverage_ratio:.2%}")
            
            # Check for auto-assignment of label
            should_auto = False
            predicted_label = None
            if hasattr(self, 'label_predictor'):
                should_auto, predicted_label = self.label_predictor.should_auto_assign(self.current_image, mask)
            
            if should_auto:
                # Auto-assign the label
                self.current_label = predicted_label
                print(f"Auto-assigned label: {predicted_label}")
            else:
                # Get predictions for suggestions
                predictions = self.label_predictor.predict_label(self.current_image, mask)
                if predictions:
                    print("Label suggestions:")
                    for label, prob in predictions:
                        print(f"- {label}: {prob:.2%}")
            
            # Update the display
            self.update_display_with_current_state()
            self.finish_button.setEnabled(True)
            self.is_expanding = False

    def prev_image(self):
        """Move to the previous image"""
        if not self.image_list or self.current_index <= 0:
            return

        # If we have started labeling (segmenter exists), ask for confirmation
        if self.segmenter is not None:
            confirmation = QMessageBox(self)
            confirmation.setWindowTitle("Previous Image")
            confirmation.setText("Do you want to save the current labeling and move to the previous image?")
            confirmation.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            response = confirmation.exec()
            
            if response == QMessageBox.StandardButton.Cancel:
                return
            elif response == QMessageBox.StandardButton.Yes:
                self.save_image(self.image_list[self.current_index])

        # Disable image interactions until Start button is pressed again
        self.image_label.interactions_enabled = False
        
        # Reset all the labeling-related variables
        self.segmenter = None
        self.expanded_masks = []
        self.overlay_image = None
        self.suggested_point = None
        self.combined_mask_overlay = None  # Clear the combined mask overlay when moving to next image
        
        # Hide the point selection buttons and enable Start button
        self.switch_button.hide()
        self.finish_button.hide()
        self.toggle_masks_button.hide()
        self.start_button.setEnabled(True)
        
        # Move to previous image
        self.current_index -= 1
        
        # Load and display the new image
        current_image_path = self.image_list[self.current_index]
        image = cv2.imread(current_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image
        self.show_image()

    def on_label_changed(self, label):
        """Handle label selection change"""
        self.current_label = label
        if self.current_mask is not None:
            # Get predictions for the current mask
            predictions = self.label_predictor.predict_label(self.image, self.current_mask)
            if predictions:
                print(f"Predictions for current mask with label '{label}':")
                for pred_label, prob in predictions:
                    print(f"- {pred_label}: {prob:.2%}")
            
            # Update UI
            self.update_image_display()

    # Add this method to the ImageViewer class
    def keyPressEvent(self, event):
        """Handle key press events"""
        from PyQt6.QtGui import QKeySequence
        from PyQt6.QtCore import Qt
        
        # Check for Ctrl+Z
        if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.current_mode == "creation" and self.is_selecting_points:
                # Remove the last placed point
                if self.negative_points and self.current_point_type == "negative":
                    removed_point = self.negative_points.pop()
                    print(f"Removed negative point at: {removed_point}")
                elif self.positive_points and self.current_point_type == "positive":
                    removed_point = self.positive_points.pop()
                    print(f"Removed positive point at: {removed_point}")
                    # If no positive points left, disable finish button
                    if not self.positive_points:
                        self.finish_button.setEnabled(False)
                else:
                    # Try the opposite type as well (more intuitive)
                    if self.negative_points:
                        removed_point = self.negative_points.pop()
                        print(f"Removed negative point at: {removed_point}")
                    elif self.positive_points:
                        removed_point = self.positive_points.pop()
                        print(f"Removed positive point at: {removed_point}")
                        # If no positive points left, disable finish button
                        if not self.positive_points:
                            self.finish_button.setEnabled(False)
                
                # Update the display
                self.update_preview_with_points()
        
        # Call the parent class handler
        super().keyPressEvent(event)

# Run the application
app = QApplication(sys.argv)
viewer = ImageViewer()
viewer.show()
sys.exit(app.exec())
