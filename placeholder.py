def start_labeling(self):
    """Generate the initial proposed point using SAM via the Segmenter."""
    if not self.image_list or self.current_index >= len(self.image_list):
        return

    wait_dialog = QDialog(self)
    wait_dialog.setWindowTitle("Generating Proposed Point")
    wait_dialog.setModal(True)
    wait_layout = QVBoxLayout()
    wait_label = QLabel("Please wait while the proposed point is generated...")
    wait_layout.addWidget(wait_label)
    progress_bar = QProgressBar(wait_dialog)
    progress_bar.setRange(0, 0)
    wait_layout.addWidget(progress_bar)
    wait_dialog.setLayout(wait_layout)
    wait_dialog.resize(400, 150)
    wait_dialog.show()

    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    current_image_path = self.image_list[self.current_index]
    image = cv2.imread(current_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    self.current_image = image  # Save the original image for later processing

    # Create the segmenter only once per image
    if self.segmenter is None:
        self.segmenter = Segmenter(self.current_image, sam2_checkpoint, model_cfg)

    self.masking_thread = MaskingThread(self.segmenter)
    self.masking_thread.result_ready.connect(lambda best_point: self.on_masking_complete(best_point, wait_dialog))
    self.masking_thread.start()