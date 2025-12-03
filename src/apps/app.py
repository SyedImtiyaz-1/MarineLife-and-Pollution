import cv2
import streamlit as st
import numpy as np
import os
import sys
import tempfile
from PIL import Image
import dark_channel_prior as dcp
import inference as inf
import time
from trashcan_pretrained_model import TrashCANPretrainedModel

# Add marine detection to path
sys.path.append('/home/imtiyaz/Desktop/MarineLife-and-Pollution/marine-detection/src')
from marine_detect.predict import combine_results
from ultralytics import YOLO

# Initialize TrashCAN model globally
@st.cache(allow_output_mutation=True)
def load_trashcan_model():
    """Load TrashCAN model and cache it"""
    try:
        model = TrashCANPretrainedModel()
        model.load_model()
        return model
    except Exception as e:
        st.error(f"Error loading TrashCAN model: {e}")
        return None

# Function to remove noise from an image
def remove_noise(image):
    processed_image, alpha_map = dcp.haze_removal(image, w_size=15, a_omega=0.95, gf_w_size=200, eps=1e-6)
    return processed_image

# Function to perform waste detection using TrashCAN model
def detect_waste_trashcan(image, trashcan_model):
    """
    Detect waste using the TrashCAN pre-trained model
    """
    try:
        if trashcan_model is None:
            return image, [], {'pollution': [], 'marine_life': [], 'other': []}
        
        # Ensure image is in correct format (uint8)
        if image.dtype != np.uint8:
            # Convert to uint8 if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Ensure image is RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Convert image to temporary file for model input
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
        cv2.imwrite(temp_path, image_bgr)
        
        # Run TrashCAN detection
        trash_detections, all_detections, results = trashcan_model.detect_trash(
            temp_path, conf_threshold=0.3, save_results=False
        )
        
        # Get annotated image
        if results:
            output_image = results[0].plot()
            # Convert BGR back to RGB for display
            if len(output_image.shape) == 3 and output_image.shape[2] == 3:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        else:
            output_image = image
        
        # Extract class names for compatibility
        trash_class_names = [d['class_name'] for d in trash_detections]
        
        # Organize detections for compatibility with existing code
        organized_detections = {
            'pollution': trash_class_names,
            'marine_life': [],  # TrashCAN model focuses on trash
            'other': []
        }
        
        # Add non-trash detections to 'other' category
        for detection in all_detections:
            if detection['class_id'] not in trashcan_model.trash_indices:
                organized_detections['other'].append(detection['class_name'])
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return output_image, trash_class_names, organized_detections
        
    except Exception as e:
        st.error(f"Error in TrashCAN detection: {str(e)}")
        return image, [], {'pollution': [], 'marine_life': [], 'other': []}

# Legacy function for backward compatibility
def detect_waste(image):
    output_image, class_names, all_detections = inf.detect(image)
    return output_image, class_names, all_detections

# Function to perform marine species detection
def detect_marine_species(image):
    """
    Detect marine species using the marine detection models.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        tuple: (annotated_image, detected_species_list)
    """
    try:
        # Model paths - adjust these paths as needed
        model_paths = [
            "/home/imtiyaz/Desktop/MarineLife-and-Pollution/marine-detection/models/FishInv.pt",
            "/home/imtiyaz/Desktop/MarineLife-and-Pollution/marine-detection/models/MegaFauna.pt"
        ]
        confs_threshold = [0.523, 0.546]
        
        # Check if models exist
        for model_path in model_paths:
            if not os.path.exists(model_path):
                st.error(f"Model not found: {model_path}")
                return image, []
        
        # Load models
        models = [YOLO(model_path) for model_path in model_paths]
        
        # Run detection
        combined_results = []
        detected_species = []
        
        for i, model in enumerate(models):
            results = model(image, conf=confs_threshold[i])
            combined_results.extend(results)
            
            # Extract detected class names
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        detected_species.append(f"{class_name} ({confidence:.2f})")
        
        # Combine results and annotate image
        annotated_image = combine_results(image, combined_results)
        
        return annotated_image, detected_species
        
    except Exception as e:
        st.error(f"Error in marine species detection: {str(e)}")
        return image, []

# Waste Detection Tab
def waste_detection_tab():
    st.header("🗑️ Underwater Waste Detection")
    st.write("Upload an image to detect underwater waste and pollution using TrashCAN AI model")
    
    # Model selection
    detection_model = st.selectbox(
        "Choose Detection Model:",
        ["TrashCAN (Recommended)", "MBARI Legacy"],
        help="TrashCAN model detects 14 types of trash with higher accuracy"
    )
    
    # Load TrashCAN model
    trashcan_model = None
    if detection_model == "TrashCAN (Recommended)":
        with st.spinner("Loading TrashCAN model..."):
            trashcan_model = load_trashcan_model()
        
        if trashcan_model:
            st.success("✅ TrashCAN model loaded successfully!")
            
            # Display model info
            with st.expander("ℹ️ TrashCAN Model Information"):
                model_info = trashcan_model.get_model_info()
                st.write(f"**Model Type:** {model_info['model_type']}")
                st.write(f"**Total Classes:** {model_info['total_classes']}")
                st.write("**Detectable Trash Types:**")
                for trash_type in model_info['trash_classes']:
                    st.write(f"• {trash_type.replace('trash_', '').replace('_', ' ').title()}")
        else:
            st.error("❌ Failed to load TrashCAN model. Using legacy detection.")
            detection_model = "MBARI Legacy"
    
    file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="waste_upload")
    
    if file is not None:
        # Read and process image
        input_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (640, 640))  # Increased resolution for better detection
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(input_image)
        
        with col2:
            st.subheader("Processed Image")
            with st.spinner("Removing noise..."):
                processed_image = remove_noise(input_image)
            st.image(processed_image, clamp=True)
        
        # Run waste detection based on selected model
        with st.spinner(f"Detecting waste with {detection_model}..."):
            if detection_model == "TrashCAN (Recommended)" and trashcan_model:
                output_image, class_names, all_detections = detect_waste_trashcan(processed_image, trashcan_model)
                st.info(f"🤖 Using TrashCAN AI model with 14 trash categories")
            else:
                output_image, class_names, all_detections = detect_waste(processed_image)
                st.info(f"🤖 Using MBARI legacy model")
        
        st.subheader("Detection Results")
        st.image(output_image)
        
        # Display detailed detection results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Trash Items", len(class_names))
        
        with col2:
            st.metric("Marine Life", len(all_detections.get('marine_life', [])))
        
        with col3:
            st.metric("Other Objects", len(all_detections.get('other', [])))
        
        # Display pollution results
        if len(class_names) == 0:
            st.success("🌊 **Clean Water!** No waste detected.")
        else:
            st.error(f"⚠️ **Waste Detected!**")
            st.write("**Detected trash items:**")
            for item in class_names:
                # Format trash names nicely
                formatted_name = item.replace('trash_', '').replace('_', ' ').title()
                st.write(f"• {formatted_name}")
        
        # Display marine life detected
        if all_detections.get('marine_life'):
            st.info(f"🐟 **Marine Life Present:** {', '.join(set(all_detections['marine_life']))}")
        
        # Display other objects
        if all_detections.get('other'):
            st.info(f"🔍 **Other Objects:** {', '.join(set(all_detections['other']))}")
        
        # Water quality assessment
        st.subheader("💧 Water Quality Assessment")
        if len(class_names) == 0 and all_detections.get('marine_life'):
            st.success("✅ **Healthy Ecosystem**: No pollution detected and marine life is thriving!")
        elif len(class_names) == 0:
            st.success("✅ **Clean Water**: No pollution detected.")
        elif all_detections.get('marine_life'):
            st.warning("⚠️ **Ecosystem at Risk**: Pollution detected but marine life still present.")
        else:
            st.error("🚨 **Polluted Water**: Waste detected with no visible marine life.")

# Marine Species Detection Tab
def marine_species_tab():
    st.header("🐟 Marine Species Detection")
    st.write("Upload an image to detect marine species including fish, sharks, turtles, and invertebrates")
    
    # Species information
    with st.expander("ℹ️ Detectable Species"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Fish & Invertebrates:**")
            st.write("• Butterfly Fish, Grouper, Parrotfish")
            st.write("• Snapper, Moray Eel, Sweet Lips")
            st.write("• Giant Clam, Urchin, Sea Cucumber")
            st.write("• Lobster, Crown of Thorns")
        with col2:
            st.write("**MegaFauna & Rare Species:**")
            st.write("• Sharks")
            st.write("• Sea Turtles")
            st.write("• Rays")
            st.write("• Humphead Wrasse")
            st.write("• Bumphead Parrotfish")
    
    file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="marine_upload")
    
    if file is not None:
        # Read and process image
        input_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(input_image)
        
        with col2:
            st.subheader("Detection Results")
            with st.spinner("Detecting marine species..."):
                annotated_image, detected_species = detect_marine_species(input_image)
            st.image(annotated_image)
        
        # Display results
        if detected_species:
            st.success(f"🐟 Marine Species Detected!")
            st.write("**Detected Species:**")
            for species in detected_species:
                st.write(f"• {species}")
        else:
            st.info("No marine species detected in this image.")

# Combined Video Detection Function
def process_video_frame(frame, waste_model=None, marine_models=None, marine_confs=[0.523, 0.546]):
    """
    Process a single video frame for both waste and marine species detection.
    
    Args:
        frame: Input video frame
        waste_model: Waste detection model (if available)
        marine_models: List of marine detection models
        marine_confs: Confidence thresholds for marine models
    
    Returns:
        tuple: (annotated_frame, detections_info)
    """
    detections_info = {
        'waste': [],
        'marine_species': [],
        'frame_number': 0
    }
    
    annotated_frame = frame.copy()
    
    try:
        # Marine species detection
        if marine_models:
            marine_results = []
            for i, model in enumerate(marine_models):
                results = model(frame, conf=marine_confs[i])
                marine_results.extend(results)
                
                # Extract species info
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            detections_info['marine_species'].append(f"{class_name} ({confidence:.2f})")
            
            # Apply marine detections to frame
            if marine_results:
                annotated_frame = combine_results(annotated_frame, marine_results)
        
        # Waste detection (simplified for video processing)
        if waste_model:
            # For video, we skip noise removal to maintain real-time performance
            waste_output, waste_classes, all_detections = inf.detect(frame)
            if waste_classes:
                detections_info['waste'] = waste_classes
                # Overlay waste detection results
                annotated_frame = waste_output
    
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
    
    return annotated_frame, detections_info

# Video Detection Tab
def video_detection_tab():
    st.header("🎥 Video Detection - Marine Life & Waste")
    st.write("Upload a video to detect both marine species and underwater waste in real-time")
    
    # Video upload
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"], key="video_upload")
    
    if uploaded_video is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Original Video")
            st.video(uploaded_video)
        
        with col2:
            st.subheader("Processing Options")
            
            # Processing options
            process_marine = st.checkbox("Detect Marine Species", value=True)
            process_waste = st.checkbox("Detect Waste", value=True)
            
            # Frame sampling
            frame_skip = st.slider("Process every Nth frame", 1, 10, 3, 
                                 help="Higher values = faster processing, lower accuracy")
            
            # Quality settings
            output_quality = st.selectbox("Output Quality", 
                                        ["High (slower)", "Medium", "Low (faster)"], 
                                        index=1)
            
            process_button = st.button("🚀 Process Video")
        
        if process_button:
            # Load models based on selection
            marine_models = None
            if process_marine:
                try:
                    model_paths = [
                        "/home/imtiyaz/Desktop/MarineLife-and-Pollution/marine-detection/models/FishInv.pt",
                        "/home/imtiyaz/Desktop/MarineLife-and-Pollution/marine-detection/models/MegaFauna.pt"
                    ]
                    
                    # Check if models exist
                    models_exist = all(os.path.exists(path) for path in model_paths)
                    if models_exist:
                        marine_models = [YOLO(path) for path in model_paths]
                        st.success("✅ Marine detection models loaded")
                    else:
                        st.warning("⚠️ Marine detection models not found. Download models first.")
                        process_marine = False
                except Exception as e:
                    st.error(f"Error loading marine models: {str(e)}")
                    process_marine = False
            
            # Process video
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Set output quality
            if output_quality == "High (slower)":
                output_width, output_height = width, height
            elif output_quality == "Medium":
                output_width, output_height = width//2, height//2
            else:  # Low
                output_width, output_width = width//4, height//4
            
            # Create output video
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps//frame_skip, (output_width, output_height))
            
            # Detection statistics
            total_detections = {
                'marine_species': set(),
                'waste_items': set(),
                'frames_processed': 0
            }
            
            frame_count = 0
            processed_frames = 0
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every Nth frame
                    if frame_count % frame_skip == 0:
                        # Resize frame if needed
                        if output_quality != "High (slower)":
                            frame = cv2.resize(frame, (output_width, output_height))
                        
                        # Process frame
                        annotated_frame, detections = process_video_frame(
                            frame, 
                            marine_models=marine_models if process_marine else None
                        )
                        
                        # Update statistics
                        total_detections['marine_species'].update(detections['marine_species'])
                        total_detections['waste_items'].update(detections['waste'])
                        total_detections['frames_processed'] += 1
                        
                        # Write frame
                        out.write(annotated_frame)
                        processed_frames += 1
                        
                        # Update progress
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{total_frames} | "
                                       f"Marine: {len(total_detections['marine_species'])} species | "
                                       f"Waste: {len(total_detections['waste_items'])} items")
                    
                    frame_count += 1
                
                # Cleanup
                cap.release()
                out.release()
                
                # Show results
                st.success("✅ Video processing completed!")
                
                # Display processed video
                st.subheader("Processed Video")
                with open(output_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                # Download button
                st.download_button(
                    label="📥 Download Processed Video",
                    data=video_bytes,
                    file_name=f"processed_marine_detection_{int(time.time())}.mp4",
                    mime="video/mp4"
                )
                
                # Show detection summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🐟 Marine Species Detected")
                    if total_detections['marine_species']:
                        for species in sorted(total_detections['marine_species']):
                            st.write(f"• {species}")
                    else:
                        st.info("No marine species detected")
                
                with col2:
                    st.subheader("🗑️ Waste Items Detected")
                    if total_detections['waste_items']:
                        for waste in sorted(total_detections['waste_items']):
                            st.write(f"• {waste}")
                    else:
                        st.info("No waste detected")
                
                # Processing statistics
                st.subheader("📊 Processing Statistics")
                st.write(f"• Total frames: {total_frames}")
                st.write(f"• Frames processed: {processed_frames}")
                st.write(f"• Processing ratio: 1 in {frame_skip} frames")
                st.write(f"• Unique marine species: {len(total_detections['marine_species'])}")
                st.write(f"• Unique waste items: {len(total_detections['waste_items'])}")
                
            except Exception as e:
                st.error(f"Error during video processing: {str(e)}")
            finally:
                # Cleanup temporary files
                try:
                    os.unlink(video_path)
                    os.unlink(output_path)
                except:
                    pass

# Main app function
def main():
    st.set_page_config(
        page_title="Marine Life & Pollution Detection",
        page_icon="🌊",
        layout="wide"
    )
    
    st.title("🌊 Marine Life & Pollution Detection System")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🗑️ Waste Detection", "🐟 Marine Species Detection", "🎥 Video Detection"])
    
    with tab1:
        waste_detection_tab()
    
    with tab2:
        marine_species_tab()
    
    with tab3:
        video_detection_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Marine Conservation Technology")

if __name__ == "__main__":
    main()
