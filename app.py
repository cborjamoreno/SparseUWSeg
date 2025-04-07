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

from segmenter_scop import Segmenter

from label_dialog import LabelDialog

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
        if self.interactions_enabled:   
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
        self.select_button = QPushButton("Select Folder", self)
        self.select_button.clicked.connect(self.select_folder)
        self.select_button.setFixedSize(150, 40)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        self.select_button.enterEvent = lambda e: self.on_cursor_over_button()

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_labeling)
        self.start_button.setFixedSize(150, 40)
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.start_button.enterEvent = lambda e: self.on_cursor_over_button()

        # Navigation buttons
        self.prev_button = QPushButton("<", self)
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setFixedSize(40, 40)
        self.prev_button.setEnabled(False)
        self.prev_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.prev_button.enterEvent = lambda e: self.on_cursor_over_button()

        self.next_button = QPushButton(">", self)
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setFixedSize(40, 40)
        self.next_button.setEnabled(False)
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.next_button.enterEvent = lambda e: self.on_cursor_over_button()

        # New buttons for point selection
        self.switch_button = QPushButton("Negative", self)
        self.switch_button.clicked.connect(self.switch_point_type)
        self.switch_button.setFixedSize(150, 40)
        self.switch_button.setEnabled(False)
        self.switch_button.setStyleSheet("""
            QPushButton {
                background-color: #E60000;
                color: white;
                border: 1px solid black;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #CC0000;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.switch_button.enterEvent = lambda e: self.on_cursor_over_button()

        self.finish_button = QPushButton("✓", self)
        self.finish_button.clicked.connect(self.finish_points)
        self.finish_button.setFixedSize(40, 40)
        self.finish_button.setEnabled(False)
        self.finish_button.setStyleSheet("""
            QPushButton {
                background-color: #006400;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #004d00;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.finish_button.enterEvent = lambda e: self.on_cursor_over_button()

        # Create a horizontal layout for the bottom buttons
        bottom_button_layout = QHBoxLayout()
        bottom_button_layout.addStretch()
        bottom_button_layout.addWidget(self.start_button)
        bottom_button_layout.addWidget(self.switch_button)
        bottom_button_layout.addWidget(self.finish_button)
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

    def select_folder(self):
        folder = "/home/cesar/LabelExpansion/Datasets/MosaicsUCSD/train/images_prueba2"
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
        self.switch_button.setStyleSheet("background-color: #808080; color: white;")  # Gray background
        self.finish_button.show()
        self.finish_button.setEnabled(False)  # Will be enabled after first positive point
        self.start_button.setEnabled(False)  # Disable Start button until Next Image is pressed
        self.is_selecting_points = True

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
        if best_point:
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

        # Add point to appropriate list
        if self.current_point_type == "positive":
            self.positive_points.append(current_point)
            print(f"Positive point added at: {current_point}")
            # Enable finish button and switch button after first positive point
            self.finish_button.setEnabled(True)
            self.switch_button.setEnabled(True)
            self.switch_button.setStyleSheet("""
                QPushButton {
                    background-color: #E60000;
                    color: white;
                    border: 1px solid black;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #CC0000;
                }
                QPushButton:disabled {
                    background-color: #BDBDBD;
                }
            """)
        else:
            self.negative_points.append(current_point)
            print(f"Negative point added at: {current_point}")

        # Start with original image and add cached overlay if it exists
        overlay_image = self.current_image.copy()
        if self.combined_mask_overlay is not None:
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
                colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray instead of green
                overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)

        # Add suggested point if it exists
        if hasattr(self, 'suggested_point') and self.suggested_point:
            row, col = self.suggested_point
            cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 0, 0), 2)
            cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 0, 0), 2)

        self.update_display(overlay_image)

    def on_mask_expansion_complete(self, mask, stored_positive_points):
        if mask is None:
            print("No mask generated.")
            return

        dialog = LabelDialog(self.labels, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            label = dialog.selected_label
            if label is None:
                label = dialog.new_label_edit.text()
                if label and dialog.chosen_color:
                    self.labels[label] = dialog.chosen_color
            color = self.labels.get(label)
            if color:
                # Keep storing in expanded_masks for future reference
                self.expanded_masks.append((mask, label, color))
                
                # Update the combined overlay for efficient display
                if self.combined_mask_overlay is None:
                    self.combined_mask_overlay = np.zeros_like(self.current_image)
                
                colored_mask = np.zeros_like(self.current_image)
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

        # Start the next labeling cycle
        QTimer.singleShot(100, self.start_labeling)

    def on_mouse_moved(self, pos):
        # Process movement immediately without timer
        if hasattr(self, 'waiting_for_negative') and self.waiting_for_negative:
            # If waiting for negative point, do two-point dynamic expansion
            self.dynamic_expand_with_negative(pos)
        else:
            # If not waiting for negative point, do normal single-point dynamic expansion
            self.dynamic_expand(pos)

    def on_cursor_over_button(self):
        """Called when cursor enters any button"""
        if self.is_selecting_points:
            # If we have points placed, show dynamic expansion with those points
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
                if self.combined_mask_overlay is not None:
                    overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.6, 0)
                
                # Add suggested point if it exists
                if hasattr(self, 'suggested_point') and self.suggested_point:
                    row, col = self.suggested_point
                    cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 0, 0), 2)
                    cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 0, 0), 2)
                
                self.update_display(overlay_image)

    def dynamic_expand(self, pos):
        if not self.is_selecting_points or self.current_image is None or self.displayed_pixmap is None:
            return

        # Check if cursor is within the image display area
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap_width = self.displayed_pixmap.width()
        pixmap_height = self.displayed_pixmap.height()

        offset_x = (label_width - pixmap_width) / 2
        offset_y = (label_height - pixmap_height) / 2

        # Start with original image and add cached overlay if it exists
        overlay_image = self.current_image.copy()
        if self.combined_mask_overlay is not None:
            overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.6, 0)

        # Draw all current points
        for point in self.positive_points:
            cv2.circle(overlay_image, point, 4, (0, 255, 0), -1)  # Green for positive
        for point in self.negative_points:
            cv2.circle(overlay_image, point, 4, (255, 0, 0), -1)  # Red for negative

        # Do dynamic expansion with cursor point if it's within the image display
        if offset_x <= pos.x() <= offset_x + pixmap_width and offset_y <= pos.y() <= offset_y + pixmap_height:
            cursor_point = self.get_image_coordinates(pos)
            if cursor_point is not None:
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
        else:
            # If cursor is outside, only show expansion with placed points if any exist
            if self.positive_points or self.negative_points:
                points = np.array(self.positive_points + self.negative_points)
                labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points))
                preview_mask = self.segmenter.propagate_points(points, labels, update_expanded_mask=False)
                if preview_mask is not None:
                    colored_preview = np.zeros_like(overlay_image)
                    colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray instead of green
                    overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)

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
            self.switch_button.setStyleSheet("""
                QPushButton {
                    background-color: #00FF00;
                    color: black;
                    border: 1px solid black;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #00E600;
                }
                QPushButton:disabled {
                    background-color: #BDBDBD;
                }
            """)
        else:
            self.current_point_type = "positive"
            self.switch_button.setText("Negative")
            self.switch_button.setStyleSheet("""
                QPushButton {
                    background-color: #E60000;
                    color: white;
                    border: 1px solid black;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #CC0000;
                }
                QPushButton:disabled {
                    background-color: #BDBDBD;
                }
            """)

    def finish_points(self):
        """Finish adding points and proceed with mask expansion"""
        if not self.positive_points:
            return

        # Store points before clearing them
        stored_positive_points = self.positive_points.copy()
        stored_negative_points = self.negative_points.copy()

        # Convert points to numpy arrays
        points = np.array(stored_positive_points + stored_negative_points)
        labels = np.array([1] * len(stored_positive_points) + [0] * len(stored_negative_points))
        
        # Reset points for next mask
        self.positive_points = []
        self.negative_points = []
        self.current_point_type = "positive"
        self.switch_button.setText("Negative")
        self.switch_button.setStyleSheet("background-color: #808080; color: white;")  # Gray background
        self.switch_button.setEnabled(False)
        self.finish_button.setEnabled(False)
        self.is_selecting_points = False

        # Process the points
        self.suggested_point = None
        self.expansion_thread = MaskExpansionThread(self.segmenter, points, labels)
        self.expansion_thread.result_ready.connect(lambda mask: self.on_mask_expansion_complete(mask, stored_positive_points))
        self.expansion_thread.start()

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
        self.start_button.setEnabled(True)
        
        # Move to previous image
        self.current_index -= 1
        
        # Load and display the new image
        current_image_path = self.image_list[self.current_index]
        image = cv2.imread(current_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image
        self.show_image()

# Run the application
app = QApplication(sys.argv)
viewer = ImageViewer()
viewer.show()
sys.exit(app.exec())
