from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QListWidget, QColorDialog, QListWidgetItem
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt

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

        # Existing labels list
        self.label_list = QListWidget()
        self.label_list.itemClicked.connect(self.on_label_selected)
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

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setEnabled(False)  # Start with OK button disabled
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(cancel_button)

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
            item.setBackground(color)
            self.label_list.addItem(item)

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
            self.selected_label = None
        else:
            self.ok_button.setEnabled(False)

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
            # Enable OK button only if we also have text
            self.ok_button.setEnabled(bool(self.new_label_edit.text()))

    def accept(self):
        """Handle OK button click"""
        if self.selected_label is None:
            # Creating new label
            new_label = self.new_label_edit.text()
            if new_label and self.chosen_color:
                self.labels[new_label] = self.chosen_color
                self.selected_label = new_label
                self.update_label_list()  # Update the list immediately
        super().accept()