from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QListWidget, QColorDialog, QListWidgetItem,
    QStyledItemDelegate, QStyle, QGridLayout
)
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtCore import Qt
import os

# Add the load_stylesheet function
def load_stylesheet(file_path):
    """Load stylesheet from a file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading stylesheet: {e}")
        return ""

class ColoredItemDelegate(QStyledItemDelegate):
    def is_dark_color(self, color):
        """Determine if a color is dark based on its luminance"""
        # Calculate relative luminance using the formula from WCAG 2.0
        r, g, b = color.red(), color.green(), color.blue()
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return luminance < 0.35  # Return True for dark colors

    def paint(self, painter, option, index):
        # Get the color from the item's data
        color = index.data(Qt.ItemDataRole.UserRole)
        
        # Adjust the rectangle if the item is selected
        rect = option.rect
        if option.state & QStyle.StateFlag.State_Selected:
            # Make the selected item slightly bigger
            rect = rect.adjusted(-2, -2, 2, 2)
        
        if color:
            # Fill the entire item with the color
            painter.fillRect(rect, color)
        
        # Determine text and border color based on background color
        is_dark = self.is_dark_color(color)
        # Use light gray for dark backgrounds, dark gray for light backgrounds
        gray_value = 200 if is_dark else 50  # 200 is light gray, 50 is dark gray
        text_color = QColor(gray_value, gray_value, gray_value)
        border_color = QColor(gray_value, gray_value, gray_value)
        
        # Draw the text with indentation
        painter.setPen(text_color)
        text_rect = rect.adjusted(10, 0, 0, 0)  # Add 10 pixels of indentation from the left
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, index.data())
        
        # Draw selection border if item is selected
        if option.state & QStyle.StateFlag.State_Selected:
            pen = painter.pen()
            pen.setWidth(5)  # Make the pen thicker
            pen.setColor(border_color)
            painter.setPen(pen)
            # Draw a thicker border with consistent width
            painter.drawRect(rect.adjusted(2, 2, -2, -2))

class LabelDialog(QDialog):
    def __init__(self, existing_labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select or Create Label")
        self.setModal(True)
        self.setMinimumWidth(300)  # Set a minimum width for better visibility

        # Load the stylesheet from this module's directory
        qss_path = os.path.join(os.path.dirname(__file__), "button_styles.qss")
        stylesheet = load_stylesheet(qss_path)
        self.setStyleSheet(stylesheet)

        # Store the labels dictionary
        self.labels = existing_labels
        self.selected_label = None
        self.chosen_color = None

        # Create widgets
        layout = QVBoxLayout()

        # Buttons with QSS classes
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setEnabled(False)  # Start with OK button disabled
        self.ok_button.setProperty("class", "start-button")  # Use blue style for OK button

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setProperty("class", "select-folder-button")  # Use neutral style for Cancel

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(cancel_button)

        # Existing labels list
        self.label_list = QListWidget()
        self.label_list.itemClicked.connect(self.on_label_selected)
        self.label_list.setItemDelegate(ColoredItemDelegate())
        self.label_list.setStyleSheet(
            """
            QListWidget::item {
                padding: 10px;
                font-size: 14px;
            }
            """
        )  # Make items bigger
        self.update_label_list()

        # New label input
        new_label_layout = QHBoxLayout()
        self.new_label_edit = QLineEdit()
        self.new_label_edit.setPlaceholderText("Enter new label name")
        self.new_label_edit.textChanged.connect(self.on_text_changed)

        self.color_button = QPushButton("Choose Color")
        self.color_button.clicked.connect(self.choose_color)
        self.color_button.setProperty("class", "select-folder-button")  # Use white style instead of orange

        new_label_layout.addWidget(self.new_label_edit)
        new_label_layout.addWidget(self.color_button)

        # Add all widgets to main layout
        layout.addWidget(QLabel("Select existing label:"))
        layout.addWidget(self.label_list)
        layout.addWidget(QLabel("Or create new label:"))
        layout.addLayout(new_label_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_label_list(self):
        """Update the list of available labels with color indicators"""
        self.label_list.clear()
        for label, color in self.labels.items():
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, color)  # Store the color in the item's data
            self.label_list.addItem(item)
        
        # Auto-select the first item if there are any items
        if self.label_list.count() > 0:
            first_item = self.label_list.item(0)
            self.label_list.setCurrentItem(first_item)
            self.selected_label = first_item.text()
            self.ok_button.setEnabled(True)

    def on_label_selected(self, item):
        """Handle selection from existing labels"""
        self.selected_label = item.text()
        self.new_label_edit.clear()  # Clear the new label input
        self.chosen_color = None
        self.color_button.setStyleSheet("")  # Reset color button appearance
        self.ok_button.setEnabled(True)

    def on_text_changed(self, text):
        """Handle changes in the new label text input"""
        if text and self.chosen_color:
            self.ok_button.setEnabled(True)
            self.label_list.clearSelection()  # Clear any selected existing label
            self.selected_label = text  # Set the selected label to the new text
        else:
            self.ok_button.setEnabled(False)
            self.selected_label = None

    def choose_color(self):
        """Open QColorDialog with customized basic colors"""
        # Create standard color dialog
        color_dialog = QColorDialog(self)
        
        # Define our distinct colors optimized for underwater imagery
        distinct_colors = [
            QColor(255, 0, 0),      # Red - good contrast underwater
            QColor(255, 165, 0),    # Orange - excellent visibility underwater
            QColor(0, 0, 255),      # Blue - still useful for some contexts
            QColor(255, 255, 0),    # Yellow - high contrast underwater
            QColor(255, 0, 255),    # Magenta - excellent contrast
            QColor(0, 255, 255),    # Cyan - moderate contrast
            QColor(255, 100, 0),    # Dark Orange - very visible
            QColor(128, 0, 255),    # Purple - good contrast
            QColor(255, 20, 147),   # Deep Pink - high visibility
            QColor(255, 69, 0),     # Red Orange - excellent for water
            QColor(255, 215, 0),    # Gold - high contrast
            QColor(220, 20, 60),    # Crimson - good visibility
            QColor(255, 140, 140),  # Light Red - softer but visible
            QColor(255, 182, 193),  # Light Pink - gentle contrast
            QColor(173, 216, 230),  # Light Blue - subtle option
        ]
        
        # Use the Qt dialog instead of native dialog
        color_dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        
        # Clear all standard colors (set to white or transparent)
        for i in range(48):
            color_dialog.setStandardColor(i, QColor(255, 255, 255, 0).rgb())
        
        # Set our distinct colors as the standard colors (consistent across all images)
        for i, color in enumerate(distinct_colors):
            if i < 48:  # Safety check
                color_dialog.setStandardColor(i, color.rgb())

        # Set default selected color to Red
        color_dialog.setCurrentColor(QColor(255, 0, 0))

        # Show the dialog
        if color_dialog.exec() == QColorDialog.DialogCode.Accepted:
            self.chosen_color = color_dialog.currentColor()
            r, g, b = self.chosen_color.red(), self.chosen_color.green(), self.chosen_color.blue()
            
            # Use a border and background to show the color
            self.color_button.setText("Color Selected")
            self.color_button.setStyleSheet(
                f"QPushButton {{background-color: rgb({r}, {g}, {b}); "
                f"border: 1px solid black; color: {'white' if self.is_dark_color(self.chosen_color) else 'black'};}}"
            )
            
            # Enable OK button and set selected label if we have text
            if self.new_label_edit.text():
                self.ok_button.setEnabled(True)
                self.selected_label = self.new_label_edit.text()
                
    def is_dark_color(self, color):
        """Determine if a color is dark based on its luminance"""
        r, g, b = color.red(), color.green(), color.blue()
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return luminance < 0.5  # Return True for dark colors

    def on_custom_color_selected(self, color, dialog):
        """Handle click on a custom color button"""
        dialog.setCurrentColor(color)
        dialog.accept()

    def accept(self):
        """Handle OK button click"""
        if self.selected_label is None:
            # Creating new label
            new_label = self.new_label_edit.text()
            if new_label and self.chosen_color:
                self.labels[new_label] = self.chosen_color
                self.selected_label = new_label
                self.update_label_list()  # Update the list immediately
                # Find and select the newly created label
                for i in range(self.label_list.count()):
                    if self.label_list.item(i).text() == new_label:
                        self.label_list.setCurrentItem(self.label_list.item(i))
                        break
        else:
            # Using existing label
            if self.selected_label not in self.labels:
                # If somehow the selected label is not in the dictionary, add it
                self.labels[self.selected_label] = self.chosen_color
        super().accept()