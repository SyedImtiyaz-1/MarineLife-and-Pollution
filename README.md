# 🌊 Neural Ocean - Marine Life and Pollution Detection

A comprehensive AI-powered system for underwater waste detection, marine life monitoring, and water quality assessment using computer vision and machine learning.

## 🎯 Project Overview

Neural Ocean addresses the critical issue of underwater pollution in oceans and seas through three integrated AI solutions:

1. **Underwater Waste Detection** - YOLOv8-based model for detecting marine debris
2. **Water Quality Assessment** - Rule-based classifier for aquatic habitat evaluation  
3. **Water Potability Testing** - ML model for drinking water safety classification

## 🚀 Features

- **Real-time Detection**: Identify 13 different classes including trash, marine life (fish, shark, ray, eel, jellyfish, crab, starfish), and other objects
- **Comprehensive Analysis**: Categorizes detections into pollution, marine life, and other categories
- **Water Quality Assessment**: Evaluates water suitability for aquatic life based on chemical properties
- **Potability Testing**: Determines if water is safe for human consumption
- **Interactive Dashboard**: Streamlit-based web interface with visualization and reporting
- **Detailed Reports**: Generate comprehensive analysis reports with charts and statistics

## 🏗️ Project Structure

```
marinelife/
├── src/
│   ├── apps/                    # Main application modules
│   │   ├── main_app.py         # Main Streamlit application
│   │   ├── app.py              # Underwater waste detection app
│   │   ├── app2.py             # Water potability testing app
│   │   ├── rule_based_classifier.py  # Water quality assessment
│   │   ├── inference.py        # MBARI model inference engine
│   │   ├── optimized_trash_detector.py
│   │   └── trashcan_pretrained_model.py
│   └── utils/                   # Utility functions
│       ├── cv2_patch.py        # OpenCV compatibility patch
│       ├── dark_channel_prior.py
│       ├── convert_trashcan_to_yolo.py
│       └── train_trashcan_yolo.py
├── notebooks/                   # Jupyter notebooks for training
│   ├── Train_TrashCAN_YoloV8.ipynb
│   ├── Train_Underwater_Waste_Detection_YoloV8.ipynb
│   └── Train_Water_Potabililty.ipynb
├── data/
│   └── models/                 # Trained models and datasets
├── assets/                     # Images and media files
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster inference)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SyedImtiyaz-1/MarineLife-and-Pollution.git
   cd MarineLife-and-Pollution
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models**
   - Place the MBARI model (`trash_mbari_09072023_640imgsz_50epochs_yolov8.pt`) in `data/models/`
   - Ensure XGBoost model (`xgboost_without_source_month.pkl`) is in `data/models/`

## 🚀 Usage

### Run the Main Application
```bash
streamlit run src/apps/main_app.py
```

### Individual Components

1. **Underwater Waste Detection**
   ```python
   from src.apps.inference import detect
   result_image, pollution_items, all_detections = detect(image_path)
   ```

2. **Water Quality Assessment**
   ```python
   from src.apps.rule_based_classifier import assess_water_quality
   quality_score = assess_water_quality(chemical_parameters)
   ```

3. **Water Potability Testing**
   ```python
   from src.apps.app2 import predict_potability
   is_potable = predict_potability(water_parameters)
   ```

## 📊 Models and Performance

### MBARI Underwater Detection Model
- **Architecture**: YOLOv8
- **Classes**: 13 (trash, marine life, objects)
- **Training**: 640px image size, 50 epochs
- **Categories**: 
  - Pollution: trash
  - Marine Life: fish, shark, ray, eel, jellyfish, crab, starfish, animal_misc
  - Other: rov, plant, shells, bird

### Water Quality Classifier
- **Method**: Rule-based classification
- **Standards**: US EPA and WHO guidelines
- **Parameters**: pH, dissolved oxygen, temperature, turbidity, etc.

### Water Potability Model
- **Algorithm**: XGBoost
- **Training Data**: 6+ million samples
- **Features**: Chemical composition analysis

## 🎨 Web Interface

The Streamlit application provides:
- **Home Dashboard**: Project overview and navigation
- **Detection Interface**: Upload images for waste detection
- **Quality Assessment**: Input water parameters for habitat analysis
- **Potability Testing**: Evaluate drinking water safety
- **Reports**: Comprehensive analysis with visualizations

## 📈 Results and Reporting

Generate detailed reports including:
- Waste detection frequency charts
- Water quality pie charts for aquatic habitat suitability
- Potability assessment results
- Trend analysis and recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MBARI (Monterey Bay Aquarium Research Institute) for the underwater detection dataset
- US EPA and WHO for water quality guidelines
- YOLOv8 by Ultralytics for object detection framework
- Streamlit for the web application framework

## 📞 Contact

**Syed Imtiyaz** - [GitHub Profile](https://github.com/SyedImtiyaz-1)

Project Link: [https://github.com/SyedImtiyaz-1/MarineLife-and-Pollution](https://github.com/SyedImtiyaz-1/MarineLife-and-Pollution)

---

🌊 **Protecting our oceans through AI-powered solutions** 🌊