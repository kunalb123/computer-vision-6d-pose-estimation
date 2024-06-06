import json
import os
from PIL import Image

def bop_to_coco(bop_root, output_file, target_object_id, use_amodal_bbox=True, train=True):
    category_map = {}
    categories = []
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    # Load category information
    models_info_path = os.path.join(bop_root, 'lm_models', 'models', 'models_info.json')
    with open(models_info_path) as f:
        models_info = json.load(f)
    for model_id, model_info in models_info.items():
        categories.append({
            "id": int(model_id),
            "name": f"obj_{int(model_id):06d}",
            "supercategory": "object"
        })
        category_map[int(model_id)] = f"obj_{int(model_id):06d}"
    
    # Determine the scenes root and image extension based on the mode
    scenes_root_train = os.path.join(bop_root, 'train')
    scenes_root_test = os.path.join(bop_root, 'test')
        
    scenes_train = sorted([d for d in os.listdir(scenes_train) if os.path.isdir(os.path.join(scenes_root_train, d))])
    scenes_test = sorted([d for d in os.listdir(scenes_test) if os.path.isdir(os.path.join(scenes_root_test, d))])
    zipped = zip(scenes_train, scenes_test)
    min_length = min(len(scenes_train), len(scenes_test))
    if train:
        SPLIT = .8
    else:
        SPLIT = .2
    stop = int(SPLIT * min_length)

    for i, out in enumerate(zipped):
        if i == stop:
            break
        scene_train, scene_test = out
        generate_im_ann(scenes_root_train, scene_train, images, annotations, use_amodal_bbox, img_id, ann_id)
        generate_im_ann(scenes_root_test, scene_test, images, annotations, use_amodal_bbox, img_id, ann_id)

    # Create COCO format dictionary
    coco_format = {
        "info": {
            "year": 2021,
            "version": "1.0",
            "description": "BOP dataset converted to COCO format",
            "date_created": "2021-01-01 00:00:00"
        },
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(coco_format, f)

def generate_im_ann(scenes_root, scene, images, annotations, use_amodal_bbox, img_id, ann_id):
    scene_path = os.path.join(scenes_root, scene)
    
    # Skip scenes that do not match the target object ID
    if int(scene) != target_object_id:
        return
    
    with open(os.path.join(scene_path, 'scene_gt.json')) as f:
        scene_gt = json.load(f)
    with open(os.path.join(scene_path, 'scene_gt_info.json')) as f:
        scene_gt_info = json.load(f)
    with open(os.path.join(scene_path, 'scene_camera.json')) as f:
        scene_camera = json.load(f)
    
    # Iterate over images in the scene

    image_files = sorted([f for f in os.listdir(os.path.join(scene_path, 'rgb')) if f.endswith('.png')])
    for image_file in image_files:
        image_id = int(image_file.split('.')[0])
        img_path = os.path.join(scene_path, 'rgb', image_file)
        img = Image.open(img_path)
        width, height = img.size
        
        images.append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": os.path.join(scene, 'rgb', image_file)
        })
        
        # Add annotations for the image
        if str(image_id) in scene_gt:
            for obj_id, (gt, gt_info, cam) in enumerate(zip(scene_gt[str(image_id)], scene_gt_info[str(image_id)], [scene_camera[str(image_id)]])):
                category_id = gt['obj_id']
                
                # Only add annotations for the target object ID
                if category_id != target_object_id:
                    continue
                
                bbox = gt_info['bbox_obj'] if use_amodal_bbox else gt_info['bbox_visib']
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "cam_R_m2c": gt['cam_R_m2c'],
                    "cam_t_m2c": gt['cam_t_m2c'],
                    "cam_K": cam['cam_K']
                })
                ann_id += 1
        
        img_id += 1


        
# Define paths and parameters
bop_root = ''  # Set your BOP root directory here
output_file = 'train_annotations_obj1.json'
target_object_id = 1  # Set the target object ID here

# Convert using amodal bounding boxes
bop_to_coco(bop_root, output_file, target_object_id, use_amodal_bbox=True, train=True)
