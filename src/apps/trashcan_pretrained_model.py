"""
TrashCAN Pre-trained Model Setup and Inference
This script downloads and uses a pre-trained YOLOv8 model trained on TrashCAN dataset
"""

import os
import requests
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class TrashCANPretrainedModel:
    def __init__(self):
        self.model = None
        self.model_path = "models/TrashCAN_Pretrained/trashcan_yolov8.pt"
        
        # TrashCAN class names (22 classes)
        self.class_names = [
            'rov', 'plant', 'animal_fish', 'animal_starfish', 'animal_shells',
            'animal_crab', 'animal_eel', 'animal_etc', 'trash_clothing', 'trash_pipe',
            'trash_bottle', 'trash_bag', 'trash_snack_wrapper', 'trash_can', 'trash_cup',
            'trash_container', 'trash_unknown_instance', 'trash_branch', 'trash_wreckage',
            'trash_tarp', 'trash_rope', 'trash_net'
        ]
        
        # Trash-specific class indices
        self.trash_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.trash_classes = [self.class_names[i] for i in self.trash_indices]
        
    def download_pretrained_model(self):
        """Download a pre-trained TrashCAN model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Try to use your existing trained model first
        existing_models = [
            "models/TrashCAN_YoloV8_Jupyter/trashcan_detection2/weights/best.pt",
            "models/TrashCAN_YoloV8/trashcan_detection/weights/best.pt",
            "models/Water_Potability/trash_mbari_09072023_640imgsz_50epochs_yolov8.pt"
        ]
        
        for model_path in existing_models:
            if os.path.exists(model_path):
                print(f"Found existing trained model: {model_path}")
                self.model_path = model_path
                return True
        
        # If no existing model, train a new one or use MBARI model
        print("No existing TrashCAN model found.")
        print("Options:")
        print("1. Use your partially trained model from Jupyter")
        print("2. Use MBARI underwater trash model")
        print("3. Train a new model from scratch")
        
        # Use MBARI model as fallback
        mbari_path = "models/Water_Potability/trash_mbari_09072023_640imgsz_50epochs_yolov8.pt"
        if os.path.exists(mbari_path):
            print(f"Using MBARI model: {mbari_path}")
            self.model_path = mbari_path
            # Update class names for MBARI model
            self.class_names = {
                0: 'trash', 1: 'eel', 2: 'rov', 3: 'starfish', 4: 'fish', 5: 'crab',
                6: 'plant', 7: 'animal_misc', 8: 'shells', 9: 'bird', 10: 'shark', 
                11: 'jellyfish', 12: 'ray'
            }
            self.trash_indices = [0]  # Only trash class in MBARI model
            return True
        
        return False
    
    def load_model(self):
        """Load the pre-trained model"""
        if not os.path.exists(self.model_path):
            if not self.download_pretrained_model():
                raise FileNotFoundError("No pre-trained model available")
        
        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("Model loaded successfully!")
        
    def detect_trash(self, image_path, conf_threshold=0.5, save_results=True):
        """Detect trash in an image"""
        if self.model is None:
            self.load_model()
        
        try:
            # Run inference
            results = self.model(image_path, conf=conf_threshold)
            
            trash_detections = []
            all_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        # Get class name
                        if isinstance(self.class_names, dict):
                            class_name = self.class_names.get(class_id, f"class_{class_id}")
                        else:
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': bbox
                        }
                        
                        all_detections.append(detection)
                        
                        # Check if it's trash
                        if class_id in self.trash_indices or 'trash' in class_name.lower():
                            trash_detections.append(detection)
            
            # Save annotated image
            if save_results and results:
                try:
                    annotated_img = results[0].plot()
                    output_path = f"trash_detection_result_{Path(image_path).stem}.jpg"
                    cv2.imwrite(output_path, annotated_img)
                    print(f"Annotated image saved: {output_path}")
                except Exception as e:
                    print(f"Warning: Could not save annotated image: {e}")
            
            return trash_detections, all_detections, results
            
        except Exception as e:
            print(f"Error in trash detection: {e}")
            return [], [], None
    
    def batch_detect(self, image_folder, conf_threshold=0.5):
        """Detect trash in multiple images"""
        if self.model is None:
            self.load_model()
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f"*{ext}"))
            image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
        
        results_summary = []
        
        for img_path in image_files:
            print(f"\nProcessing: {img_path.name}")
            trash_detections, all_detections, _ = self.detect_trash(
                str(img_path), conf_threshold, save_results=True
            )
            
            result = {
                'image': img_path.name,
                'trash_count': len(trash_detections),
                'total_detections': len(all_detections),
                'trash_types': [d['class_name'] for d in trash_detections]
            }
            results_summary.append(result)
            
            print(f"  Trash detected: {len(trash_detections)}")
            print(f"  Total objects: {len(all_detections)}")
            if trash_detections:
                print(f"  Trash types: {', '.join(set(d['class_name'] for d in trash_detections))}")
        
        return results_summary
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            self.load_model()
        
        info = {
            'model_path': self.model_path,
            'model_type': 'YOLOv8',
            'classes': self.class_names,
            'trash_classes': self.trash_classes if isinstance(self.class_names, list) else ['trash'],
            'total_classes': len(self.class_names)
        }
        
        return info

def main():
    """Example usage of TrashCAN pre-trained model"""
    
    # Initialize the model
    trash_model = TrashCANPretrainedModel()
    
    # Load the model
    try:
        trash_model.load_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have a trained model available.")
        return
    
    # Get model information
    model_info = trash_model.get_model_info()
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Model path: {model_info['model_path']}")
    print(f"Model type: {model_info['model_type']}")
    print(f"Total classes: {model_info['total_classes']}")
    print(f"Trash classes: {model_info['trash_classes']}")
    
    # Test on sample images if available
    test_folders = [
        "transcan_dataset/yolo_format/images/val",
        "assets/images",
        "test_images"
    ]
    
    for folder in test_folders:
        if os.path.exists(folder):
            print(f"\n" + "="*50)
            print(f"TESTING ON IMAGES IN: {folder}")
            print("="*50)
            
            # Process first 3 images
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(Path(folder).glob(f"*{ext}"))
                if len(image_files) >= 3:
                    break
            
            for img_path in image_files[:3]:
                print(f"\nTesting on: {img_path.name}")
                trash_detections, all_detections, _ = trash_model.detect_trash(
                    str(img_path), conf_threshold=0.3
                )
                
                print(f"  Found {len(trash_detections)} trash objects")
                print(f"  Total objects detected: {len(all_detections)}")
                
                if trash_detections:
                    for detection in trash_detections:
                        print(f"    - {detection['class_name']}: {detection['confidence']:.3f}")
            break
    
    print(f"\n" + "="*50)
    print("USAGE EXAMPLES")
    print("="*50)
    print("# Single image detection:")
    print("trash_detections, all_detections, results = trash_model.detect_trash('image.jpg')")
    print("\n# Batch processing:")
    print("results = trash_model.batch_detect('image_folder/')")
    print("\n# Get only trash detections:")
    print("trash_only = [d for d in all_detections if d['class_id'] in trash_model.trash_indices]")

if __name__ == "__main__":
    main()
