from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_trashcan_model():
    """Train YOLOv8 model on TrashCAN dataset"""
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Dataset configuration
    data_yaml = "/home/imtiyaz/Desktop/MarineLife-and-Pollution/transcan_dataset/yolo_format/data.yaml"
    
    # Create models directory if it doesn't exist
    models_dir = "/home/imtiyaz/Desktop/MarineLife-and-Pollution/models/TrashCAN_YoloV8"
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # Start with nano model for faster training
    
    # Training parameters
    training_params = {
        'data': data_yaml,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'project': models_dir,
        'name': 'trashcan_detection',
        'save_period': 10,  # Save checkpoint every 10 epochs
        'patience': 20,     # Early stopping patience
        'workers': 4,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    print("Starting YOLOv8 training on TrashCAN dataset...")
    print(f"Training parameters: {training_params}")
    
    # Train the model
    results = model.train(**training_params)
    
    # Validate the model
    print("\nValidating trained model...")
    validation_results = model.val()
    
    # Export the model
    print("\nExporting trained model...")
    model.export(format='onnx')
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {models_dir}/trashcan_detection")
    print(f"Best weights: {models_dir}/trashcan_detection/weights/best.pt")
    
    return results

def create_trash_only_model():
    """Create a version that only detects trash classes"""
    
    print("\nCreating trash-only detection script...")
    
    trash_only_script = '''
import cv2
from ultralytics import YOLO
import numpy as np

class TrashOnlyDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # TrashCAN class names
        self.all_classes = [
            'rov', 'plant', 'animal_fish', 'animal_starfish', 'animal_shells',
            'animal_crab', 'animal_eel', 'animal_etc', 'trash_clothing', 'trash_pipe',
            'trash_bottle', 'trash_bag', 'trash_snack_wrapper', 'trash_can', 'trash_cup',
            'trash_container', 'trash_unknown_instance', 'trash_branch', 'trash_wreckage',
            'trash_tarp', 'trash_rope', 'trash_net'
        ]
        
        # Only trash classes (indices 8-21)
        self.trash_class_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.trash_classes = [self.all_classes[i] for i in self.trash_class_indices]
        
    def detect_trash_only(self, image, conf_threshold=0.5):
        """Detect only trash objects in the image"""
        results = self.model(image)
        
        trash_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Only keep trash detections
                    if class_id in self.trash_class_indices and confidence >= conf_threshold:
                        trash_detections.append({
                            'class_id': class_id,
                            'class_name': self.all_classes[class_id],
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        })
        
        return trash_detections
    
    def visualize_trash_detections(self, image, detections):
        """Draw bounding boxes around trash detections"""
        img_copy = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return img_copy

# Example usage:
if __name__ == "__main__":
    # Initialize detector with trained model
    detector = TrashOnlyDetector("models/TrashCAN_YoloV8/trashcan_detection/weights/best.pt")
    
    # Load and process image
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)
    
    # Detect trash
    trash_detections = detector.detect_trash_only(image, conf_threshold=0.5)
    
    print(f"Found {len(trash_detections)} trash objects:")
    for detection in trash_detections:
        print(f"  - {detection['class_name']}: {detection['confidence']:.2f}")
    
    # Visualize results
    result_image = detector.visualize_trash_detections(image, trash_detections)
    cv2.imshow("Trash Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
    
    with open("/home/imtiyaz/Desktop/MarineLife-and-Pollution/trash_only_detector.py", "w") as f:
        f.write(trash_only_script)
    
    print("Created trash_only_detector.py for trash-specific detection")

if __name__ == "__main__":
    # Train the model
    results = train_trashcan_model()
    
    # Create trash-only detector
    create_trash_only_model()
    
    print("\nTraining and setup complete!")
    print("You can now use the trained model for trash detection.")
