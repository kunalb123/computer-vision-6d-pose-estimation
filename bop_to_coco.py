import json
import os
from PIL import Image
import random

def bop_to_coco(bop_root, output_files, target_object_id, use_amodal_bbox=True, train=True):
    category_map = {}
    categories = []
    im_ann = {'images_train' : [], 'annotations_train' : [],
              'images_test' : [], 'annotations_test' : []}
    im_ann_id = {'im_id_train': 1, 'ann_id_train': 1,
                 'im_id_test': 1, 'ann_id_test': 1}
    split = .8
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
    scene_train = os.path.join(bop_root, 'train', f'{target_object_id:06}')
    scene_test = os.path.join(bop_root, 'test', f'{target_object_id:06}')
    generate_im_ann(scene_train, im_ann, use_amodal_bbox, im_ann_id, split)
    generate_im_ann(scene_test, im_ann, use_amodal_bbox, im_ann_id, split)

    # Create COCO format dictionary
    coco_format_train = {
        "info": {
            "year": 2021,
            "version": "1.0",
            "description": "BOP dataset converted to COCO format",
            "date_created": "2021-01-01 00:00:00"
        },
        "images": im_ann['images_train'],
        "annotations": im_ann['annotations_train'],
        "categories": categories
    }
    print(len(im_ann['images_train']), len(im_ann['annotations_train']))
    coco_format_test = {
        "info": {
            "year": 2021,
            "version": "1.0",
            "description": "BOP dataset converted to COCO format",
            "date_created": "2021-01-01 00:00:00"
        },
        "images": im_ann['images_test'],
        "annotations": im_ann['annotations_test'],
        "categories": categories
    }
    print(len(im_ann['images_test']), len(im_ann['annotations_test']))
    # Save to output file
    with open(output_files['train'], 'w') as f:
        json.dump(coco_format_train, f)
    with open(output_files['test'], 'w') as f:
        json.dump(coco_format_test, f)

def generate_im_ann(scene, im_ann, use_amodal_bbox, im_ann_id, split):
    
    with open(os.path.join(scene, 'scene_gt.json')) as f:
        scene_gt = json.load(f)
    with open(os.path.join(scene, 'scene_gt_info.json')) as f:
        scene_gt_info = json.load(f)
    with open(os.path.join(scene, 'scene_camera.json')) as f:
        scene_camera = json.load(f)
    
    # Iterate over images in the scene

    image_files = [f for f in os.listdir(os.path.join(scene, 'rgb')) if f.endswith('.png')]
    random.shuffle(image_files)
    for i, image_file in enumerate(image_files):
        split_idx = int(split * len(image_files))
        if i < split_idx:
            split_name = 'train'
        else:
            split_name = 'test'
        image_id = int(image_file.split('.')[0])
        img_path = os.path.join(scene, 'rgb', image_file)
        img = Image.open(img_path)
        width, height = img.size

        
        im_ann[f'images_{split_name}'].append({
            "id": im_ann_id[f'im_id_{split_name}'],
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
                im_ann[f'annotations_{split_name}'].append({
                    "id": im_ann_id[f'ann_id_{split_name}'],
                    "image_id": im_ann_id[f'im_id_{split_name}'],
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "cam_R_m2c": gt['cam_R_m2c'],
                    "cam_t_m2c": gt['cam_t_m2c'],
                    "cam_K": cam['cam_K']
                })
                im_ann_id[f'ann_id_{split_name}'] += 1
        
        im_ann_id[f'im_id_{split_name}'] += 1


        
# Define paths and parameters
bop_root = ''  # Set your BOP root directory here
output_files = {'train': 'train_annotations_obj1.json', 'test': 'test_annotations_obj1.json'}
target_object_id = 1  # Set the target object ID here

# Convert using amodal bounding boxes
bop_to_coco(bop_root, output_files, target_object_id, use_amodal_bbox=True, train=True)
