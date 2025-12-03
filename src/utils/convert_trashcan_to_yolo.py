import json
import os
from pathlib import Path
import shutil

def convert_coco_to_yolo(coco_json_path, images_dir, output_labels_dir, output_images_dir):
    """Convert COCO format annotations to YOLO format"""
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_to_id = {cat['name']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    print(f"Found {len(categories)} categories:")
    for idx, (cat_id, cat_name) in enumerate(categories.items()):
        print(f"  {idx}: {cat_name}")
    
    # Create image mapping
    images = {img['id']: img for img in coco_data['images']}
    
    # Process annotations
    processed_images = set()
    
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        image_info = images[image_id]
        
        image_filename = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']
        
        # Copy image to output directory if not already done
        if image_id not in processed_images:
            src_image_path = os.path.join(images_dir, image_filename)
            dst_image_path = os.path.join(output_images_dir, image_filename)
            
            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dst_image_path)
                processed_images.add(image_id)
            else:
                print(f"Warning: Image {src_image_path} not found")
                continue
        
        # Convert bbox from COCO to YOLO format
        bbox = annotation['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        
        # Convert to YOLO format (normalized center coordinates)
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        width_norm = w / image_width
        height_norm = h / image_height
        
        # Get class ID (YOLO uses 0-based indexing)
        category_name = categories[annotation['category_id']]
        class_id = category_to_id[category_name]
        
        # Create label file
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(output_labels_dir, label_filename)
        
        # Append to label file (multiple objects per image)
        with open(label_path, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    
    print(f"Processed {len(processed_images)} images")
    return categories, category_to_id

def create_data_yaml(categories, output_path):
    """Create data.yaml file for YOLO training"""
    
    # Filter only trash classes for trash-only detection
    trash_classes = [name for name in categories.values() if 'trash' in name.lower()]
    
    yaml_content = f"""# TrashCAN Dataset Configuration
path: {os.path.dirname(output_path)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: {len(categories)}  # number of classes
names: {list(categories.values())}  # class names

# Trash-only classes (for reference)
trash_classes: {trash_classes}
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created data.yaml with {len(categories)} classes")

def main():
    # Paths
    base_dir = "/home/imtiyaz/Desktop/MarineLife-and-Pollution/transcan_dataset"
    dataset_dir = os.path.join(base_dir, "dataset/instance_version")
    
    # Input paths
    train_json = os.path.join(dataset_dir, "instances_train_trashcan.json")
    val_json = os.path.join(dataset_dir, "instances_val_trashcan.json")
    train_images = os.path.join(dataset_dir, "train")
    val_images = os.path.join(dataset_dir, "val")
    
    # Output paths
    output_dir = os.path.join(base_dir, "yolo_format")
    train_labels_out = os.path.join(output_dir, "labels/train")
    val_labels_out = os.path.join(output_dir, "labels/val")
    train_images_out = os.path.join(output_dir, "images/train")
    val_images_out = os.path.join(output_dir, "images/val")
    
    print("Converting TrashCAN dataset to YOLO format...")
    
    # Convert training set
    print("\nProcessing training set...")
    categories, category_mapping = convert_coco_to_yolo(
        train_json, train_images, train_labels_out, train_images_out
    )
    
    # Convert validation set
    print("\nProcessing validation set...")
    convert_coco_to_yolo(
        val_json, val_images, val_labels_out, val_images_out
    )
    
    # Create data.yaml
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    create_data_yaml(categories, data_yaml_path)
    
    print(f"\nConversion complete! YOLO dataset saved to: {output_dir}")
    print(f"Data configuration saved to: {data_yaml_path}")

if __name__ == "__main__":
    main()
