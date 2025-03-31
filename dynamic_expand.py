import numpy as np
import cv2
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

def on_mouse_moved(self, pos):
    # Each time the mouse moves, restart the timer.
    self.last_cursor_pos = pos
    self.dynamic_timer.start()

def dynamic_expand(self):
    # This function is called when the cursor stops moving for 200ms.
    if self.current_image is None or self.displayed_pixmap is None:
        return

    # Convert the mouse position to original image coordinates (same as in on_image_clicked)
    label_width = self.image_label.width()
    label_height = self.image_label.height()
    pixmap_width = self.displayed_pixmap.width()
    pixmap_height = self.displayed_pixmap.height()

    offset_x = (label_width - pixmap_width) / 2
    offset_y = (label_height - pixmap_height) / 2

    pos = self.last_cursor_pos
    if not (offset_x <= pos.x() <= offset_x + pixmap_width and offset_y <= pos.y() <= offset_y + pixmap_height):
        return

    original_h, original_w, _ = self.current_image.shape
    ratio_x = original_w / pixmap_width
    ratio_y = original_h / pixmap_height

    orig_x = int((pos.x() - offset_x) * ratio_x)
    orig_y = int((pos.y() - offset_y) * ratio_y)
    user_point = (orig_x, orig_y)

    # Get a dynamic preview mask from the segmenter without saving it permanently.
    # For example, call propagate_points on the segmenter with the current point.
    points = np.array([user_point])
    labels = np.array([1])
    preview_mask = self.segmenter.propagate_points(points, labels)

    # Create a temporary overlay that includes all permanent masks (if any)
    # plus the dynamic preview mask.
    overlay_image = self.current_image.copy()
    for m, lab, col in self.expanded_masks:
        colored_mask = np.zeros_like(overlay_image)
        colored_mask[m > 0] = [col.red(), col.green(), col.blue()]
        overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_mask, 0.6, 0)

    # Overlay the dynamic preview mask (using a distinct transparency)
    if preview_mask is not None:
        preview_color = (0, 255, 0)  # For example, green for dynamic preview.
        colored_preview = np.zeros_like(overlay_image)
        colored_preview[preview_mask > 0] = preview_color
        overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)

    # Update the display without saving the preview mask.
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