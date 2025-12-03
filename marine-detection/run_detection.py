from src.marine_detect.predict import predict_on_images, predict_on_video
import os

# Create necessary directories if they don't exist
os.makedirs("assets/images/input_folder", exist_ok=True)
os.makedirs("assets/images/output_folder", exist_ok=True)

# For images
predict_on_images(
    model_paths=["models/FishInv.pt", "models/MegaFauna.pt"],
    confs_threshold=[0.523, 0.546],
    images_input_folder_path="assets/images/input_folder",
    images_output_folder_path="assets/images/output_folder",
)