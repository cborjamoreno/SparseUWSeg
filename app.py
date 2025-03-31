import sys
import os
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QMessageBox, QDialog, QProgressBar, QLineEdit, QListWidget,
    QListWidgetItem, QColorDialog
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from segmenter import Segmenter

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
        mask = self.segmenter.propagate_points(self.points, self.labels)
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

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_labeling)
        self.start_button.setFixedSize(150, 40)
        self.start_button.setEnabled(False)

        self.next_button = QPushButton("Next Image", self)
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setFixedSize(200, 40)
        self.next_button.setEnabled(False)

        # Layout for buttons (centered horizontally)
        button_row_layout = QHBoxLayout()
        button_row_layout.addStretch()
        button_row_layout.addWidget(self.start_button)
        button_row_layout.addWidget(self.next_button)
        button_row_layout.addStretch()

        # Main layout: image on top, buttons below
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(button_row_layout)
        self.setLayout(main_layout)

        # Variables
        self.image_list = []
        self.current_index = 0
        self.current_image = None  # Original image (RGB, NumPy array)
        self.overlay_image = None # Image with all expanded masks
        self.displayed_pixmap = None
        self.segmenter = None  # Segmenter instance
        self.labels = {}  # Dictionary to store label names and QColor objects
        self.expanded_masks = [] # List to store tuples: (mask, label, color)
        self.combined_mask_overlay = None # For efficient display

        self.image_label.mouse_moved.connect(self.on_mouse_moved)

        self.waiting_for_negative = False
        self.positive_point = None

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

    def start_labeling(self):
        if not self.image_list or self.current_index >= len(self.image_list):
            return
        
        # Enable image interactions
        self.image_label.interactions_enabled = True

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
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        current_image_path = self.image_list[self.current_index]

        image = cv2.imread(current_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image

        if self.segmenter is None:
            self.segmenter = Segmenter(self.current_image, sam2_checkpoint, model_cfg)

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
        # User will now click on the image to expand the mask

    def on_image_clicked(self, pos):
        if self.current_image is None or self.displayed_pixmap is None:
            return

        # Convert click position to image coordinates
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap_width = self.displayed_pixmap.width()
        pixmap_height = self.displayed_pixmap.height()

        offset_x = (label_width - pixmap_width) / 2
        offset_y = (label_height - pixmap_height) / 2

        click_x = pos.x()
        click_y = pos.y()

        if not (offset_x <= click_x <= offset_x + pixmap_width and offset_y <= click_y <= offset_y + pixmap_height):
            print("Click outside the image area.")
            return

        original_h, original_w, _ = self.current_image.shape
        ratio_x = original_w / pixmap_width
        ratio_y = original_h / pixmap_height

        orig_x = int((click_x - offset_x) * ratio_x)
        orig_y = int((click_y - offset_y) * ratio_y)
        current_point = (orig_x, orig_y)

        if not self.waiting_for_negative:
            # This is the first click (positive point)
            self.positive_point = current_point
            self.waiting_for_negative = True
            print(f"Positive point selected at: {current_point}")
            
            # Show the positive point on the image
            temp_image = self.overlay_image.copy() if self.overlay_image is not None else self.current_image.copy()
            cv2.circle(temp_image, (orig_x, orig_y), 5, (0, 255, 0), -1)  # Green circle for positive
            self.update_display(temp_image)
            
        else:
            # This is the second click (negative point)
            print(f"Negative point selected at: {current_point}")
            self.waiting_for_negative = False
            
            # Show both points before processing
            temp_image = self.overlay_image.copy() if self.overlay_image is not None else self.current_image.copy()
            first_x, first_y = self.positive_point  # Unpack the stored point correctly
            cv2.circle(temp_image, (first_x, first_y), 5, (0, 255, 0), -1)  # Green for positive
            cv2.circle(temp_image, (orig_x, orig_y), 5, (255, 0, 0), -1)  # Red for negative
            self.update_display(temp_image)

            # Process both points
            points = np.array([self.positive_point, current_point])
            labels = np.array([1, 0])  # 1 for positive, 0 for negative
            
            self.suggested_point = None
            self.expansion_thread = MaskExpansionThread(self.segmenter, points, labels)
            self.expansion_thread.result_ready.connect(self.on_mask_expansion_complete)
            self.expansion_thread.start()


    def on_mask_expansion_complete(self, mask):
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
                
                # Make sure to get a new suggested point after adding a mask
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

    
    def dynamic_expand(self, pos):
        if self.current_image is None or self.displayed_pixmap is None:
            return

        cursor_point = self.get_image_coordinates(pos)
        if cursor_point is None:
            return

        # Single point expansion
        points = np.array([cursor_point])
        labels = np.array([1])
        
        preview_mask = self.segmenter.propagate_points(points, labels)

        # Start with original image and add cached overlay if it exists
        overlay_image = self.current_image.copy()
        if self.combined_mask_overlay is not None:
            overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.6, 0)
        
        # Add the preview mask
        if preview_mask is not None:
            colored_preview = np.zeros_like(overlay_image)
            colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray instead of green
            overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)

        # Add suggested point if it exists - moved after all overlays
        if hasattr(self, 'suggested_point') and self.suggested_point:
            row, col = self.suggested_point
            # Draw blue cross
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
        points = np.array([self.positive_point, cursor_point])
        labels = np.array([1, 0])
        
        preview_mask = self.segmenter.propagate_points(points, labels)
        
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
        pos_x, pos_y = self.positive_point
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

# Run the application
app = QApplication(sys.argv)
viewer = ImageViewer()
viewer.show()
sys.exit(app.exec())
