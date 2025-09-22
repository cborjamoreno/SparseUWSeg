from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt

class OverlapDialog(QDialog):
    def __init__(self, suggested_point, overlapping_mask_label, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Point Overlap Detected")
        self.setModal(True)
        self.result_choice = None
        
        layout = QVBoxLayout()
        
        # Title and description
        title_label = QLabel("The suggested point overlaps with an existing mask!")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        # Details
        point_y, point_x = suggested_point
        details_label = QLabel(f"Point: ({point_x}, {point_y})\nOverlapping mask: '{overlapping_mask_label}'")
        details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(details_label)
        
        # Question
        question_label = QLabel("Is this point part of the same object?")
        question_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        question_label.setStyleSheet("font-size: 12px; margin: 10px 0;")
        layout.addWidget(question_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.same_object_button = QPushButton("Same Object\n(Union masks)")
        self.same_object_button.clicked.connect(self.choose_same_object)
        self.same_object_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(self.same_object_button)
        
        self.different_object_button = QPushButton("Different Object\n(Resolve overlap)")
        self.different_object_button.clicked.connect(self.choose_different_object)
        self.different_object_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        button_layout.addWidget(self.different_object_button)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.resize(350, 200)
    
    def choose_same_object(self):
        self.result_choice = "same_object"
        self.accept()
    
    def choose_different_object(self):
        self.result_choice = "different_object"
        self.accept()
