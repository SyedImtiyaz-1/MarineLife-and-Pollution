from ..utils import cv2_patch  # Apply OpenCV compatibility patch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
import torch
import cv2

# MBARI model class labels
labels = {0: 'trash', 1: 'eel', 2: 'rov', 3: 'starfish', 4: 'fish', 5: 'crab', 
          6: 'plant', 7: 'animal_misc', 8: 'shells', 9: 'bird', 10: 'shark', 11: 'jellyfish', 12: 'ray'}

# Define pollution indicators
pollution_classes = ['trash']
marine_life_classes = ['eel', 'starfish', 'fish', 'crab', 'shark', 'jellyfish', 'ray', 'animal_misc']

garbage = []
def detect(image):
    model = YOLO("../../data/models/trash_mbari_09072023_640imgsz_50epochs_yolov8.pt")
    results = model(image)
    class_list = []
    detected_objects = {'pollution': [], 'marine_life': [], 'other': []}
    
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        if boxes is not None:
            class_list = boxes.cls.tolist()
    
    # Categorize detections
    for class_id in class_list:
        class_name = labels[int(class_id)]
        if class_name in pollution_classes:
            detected_objects['pollution'].append(class_name)
        elif class_name in marine_life_classes:
            detected_objects['marine_life'].append(class_name)
        else:
            detected_objects['other'].append(class_name)
    
    # Return only pollution items for waste detection compatibility
    pollution_items = detected_objects['pollution']
    garbage.extend(pollution_items)
    
    res_plotted = results[0].plot()
    return res_plotted, pollution_items, detected_objects

# cv2.imshow('res', res_plotted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()