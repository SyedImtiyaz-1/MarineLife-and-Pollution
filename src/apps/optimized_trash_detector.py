
from ultralytics import YOLO
import cv2

class OptimizedTrashDetector:
    def __init__(self):
        # Try underwater waste model first, fallback to MBARI
        try:
            self.model = YOLO("models/Underwater_Waste_Detection_YoloV8/60_epochs_denoised.pt")
            self.model_type = "underwater_waste"
            print("Using Underwater Waste Detection Model")
        except:
            self.model = YOLO("models/Water_Potability/trash_mbari_09072023_640imgsz_50epochs_yolov8.pt")
            self.model_type = "mbari"
            print("Using MBARI Trash Detection Model")
        
        self.class_names = self.model.names
        
    def detect_trash(self, image, conf_threshold=0.5):
        """Detect trash in image"""
        results = self.model(image, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names[class_id]
                    
                    # Filter for trash-related classes
                    if self.is_trash_class(class_name):
                        detections.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()
                        })
        
        return detections
    
    def is_trash_class(self, class_name):
        """Check if class is trash-related"""
        trash_keywords = ['trash', 'waste', 'garbage', 'litter', 'debris', 
                         'bottle', 'bag', 'can', 'container', 'wrapper']
        
        class_lower = class_name.lower()
        return any(keyword in class_lower for keyword in trash_keywords)
    
    def visualize_detections(self, image, detections):
        """Draw bounding boxes on image"""
        img_copy = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return img_copy

# Example usage
if __name__ == "__main__":
    detector = OptimizedTrashDetector()
    
    # Test with sample image if available
    import os
    if os.path.exists("assets/yacht.jpg"):
        image = cv2.imread("assets/yacht.jpg")
        detections = detector.detect_trash(image)
        
        print(f"Found {len(detections)} trash objects:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")
        
        # Show result
        result_img = detector.visualize_detections(image, detections)
        cv2.imwrite("trash_detection_result.jpg", result_img)
        print("Result saved as trash_detection_result.jpg")
