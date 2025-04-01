from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QListWidget, QColorDialog, QListWidgetItem,
    QStyledItemDelegate, QStyle
)
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtCore import Qt

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

        # Store the labels dictionary
        self.labels = existing_labels
        self.selected_label = None
        self.chosen_color = None

        # Create widgets
        layout = QVBoxLayout()

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setEnabled(False)  # Start with OK button disabled
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(cancel_button)

        # Existing labels list
        self.label_list = QListWidget()
        self.label_list.itemClicked.connect(self.on_label_selected)
        self.label_list.setItemDelegate(ColoredItemDelegate())
        self.label_list.setStyleSheet("""
            QListWidget::item { 
                padding: 10px;
                font-size: 14px;
            }
        """)  # Make items bigger
        self.update_label_list()

        # New label input
        new_label_layout = QHBoxLayout()
        self.new_label_edit = QLineEdit()
        self.new_label_edit.setPlaceholderText("Enter new label name")
        self.new_label_edit.textChanged.connect(self.on_text_changed)
        self.color_button = QPushButton("Choose Color")
        self.color_button.clicked.connect(self.choose_color)
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
        """Open color picker dialog with distinct predefined colors"""
        color_dialog = QColorDialog(self)
        
        # Define a set of distinct colors (RGB values)
        distinct_colors = [
            QColor(255, 0, 0),      # Red
            QColor(0, 255, 0),      # Green
            QColor(0, 0, 255),      # Blue
            QColor(255, 255, 0),    # Yellow
            QColor(255, 0, 255),    # Magenta
            QColor(0, 255, 255),    # Cyan
            QColor(255, 128, 0),    # Orange
            QColor(128, 0, 255),    # Purple
            QColor(0, 255, 128),    # Spring Green
            QColor(255, 0, 128),    # Rose
            QColor(128, 255, 0),    # Lime
            QColor(0, 128, 255),    # Sky Blue
            QColor(255, 128, 128),  # Light Red
            QColor(128, 255, 128),  # Light Green
            QColor(128, 128, 255),  # Light Blue
        ]
        
        # Set the custom color options
        color_dialog.setCustomColor(0, distinct_colors[0])
        for i, color in enumerate(distinct_colors):
            color_dialog.setCustomColor(i, color)
        
        # Show the dialog and get the selected color
        if color_dialog.exec() == QColorDialog.DialogCode.Accepted:
            self.chosen_color = color_dialog.selectedColor()
            self.color_button.setStyleSheet(
                f"background-color: rgb({self.chosen_color.red()}, {self.chosen_color.green()}, {self.chosen_color.blue()})"
            )
            # Enable OK button and set selected label if we have text
            if self.new_label_edit.text():
                self.ok_button.setEnabled(True)
                self.selected_label = self.new_label_edit.text()

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