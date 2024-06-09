import numpy as np
from tqdm import tqdm
from dataloader import *
from train import load_model
import torch
from model import DeepPose
import util
import yaml
from loss import CompositeLoss
import cv2
import random
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from object_detector import ObjectDetector
import os


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    device = torch.device('mps')
print('device being used:', device)

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

config_detect = lambda: None
config_detect.mask_edges = 1
config_detect.mask_faces = 1
config_detect.vertex = 1
config_detect.threshold = 0.5
config_detect.softmax = 1000
config_detect.thresh_angle = config["thresh_angle"]
config_detect.thresh_map = config["thresh_map"]
config_detect.sigma = config["sigma"]
config_detect.thresh_points = config["thresh_points"]

def compute_add(vertices, pred_pose, gt_pose):
    # Transform vertices with ground truth pose
    vertices_gt = (gt_pose[:3, :3] @ vertices + gt_pose[:3, 3:4])
    
    # Transform vertices with predicted pose
    vertices_pred = (pred_pose[:3, :3] @ vertices + pred_pose[:3, 3:4])
    
    # Compute Euclidean distances
    distances = np.linalg.norm(vertices_gt - vertices_pred, axis=1)
    
    # Return average distance
    return np.mean(distances)


def evaluate(model, dataset, obj_model, model_name='', pic_samples=10):
    
    sample_indexes = random.sample(range(len(dataset)), pic_samples)

    cuboid1 = Cuboid3d([obj_model['size_x'], obj_model['size_y'], obj_model['size_z']])
    K = util.get_cam_matrix(file='lm/camera.json')
    solver = CuboidPNPSolver(camera_intrinsic_matrix=K, cuboid3d=cuboid1)
    model.eval()
    model.to(device)
    with torch.no_grad():
        vertices = util.create_3D_vertices(obj_model)
        model_points = np.concatenate((vertices, np.mean(vertices, axis=1).reshape(3, -1)), axis=1)
        running_add = 0
        num = 0
        for i in tqdm(range(len(dataset))):
            num += 1
            print('num')
            img_norm, target_dict = dataset[i]

            detected_objects, im_belief = ObjectDetector().detect_object_in_image(
                model, solver, target_dict['img'], config_detect)
            print('HERE', detected_objects)
            for detected_object in detected_objects:
                rvec, tvec = cv2.Rodrigues(detected_object['rvec'])[0], np.array(detected_object['location']).reshape(3, -1)
                pred_pose = np.concatenate((rvec, tvec), axis=1)
                gt_rvec = np.array(target_dict['targets']['cam_R_m2c']).reshape(3, 3)
                gt_tvec = np.array(target_dict['targets']['cam_t_m2c']).reshape(3, -1)
                gt_pose = np.concatenate((gt_rvec, gt_tvec), axis=1)
                
                if i in sample_indexes:
                    util.plot_object_point_map(detected_object, 
                        target_dict['projected_vertices'], 
                        target_dict['img'], 
                        save_dir='eval_pics',
                        model_name=model_name,
                        name=i)

                add = compute_add(model_points, pred_pose, gt_pose)
                print('ADD IS:', add)
                running_add += add
    
    return running_add / num


def load_and_evaluate_model(model_checkpoint, dataset, pic_samples=10):
    model = DeepPose(
        extra_conv=util.get_info_from_model_file(model_checkpoint, 'extra_conv'),
        num_final_stages=util.get_info_from_model_file(model_checkpoint, 'stages')
    )
    load_model(model_checkpoint, model)
    score = evaluate(model=model,
                     dataset=dataset, 
                     obj_model=dataset.models['1'], 
                     model_name=util.get_info_from_model_file(model_checkpoint, 'name'), 
                     pic_samples=pic_samples)
    print('score is:', model_checkpoint, score)
    return score


def load_and_evaluate_models(model_checkpoints_root, dataset, write_file='results.log', pic_samples=10):
    model_checkpoints = [os.path.join(model_checkpoints_root, f) for f in os.listdir(model_checkpoints_root)]
    print('evaluating the following models:', model_checkpoints)
    with open(write_file, 'w') as f:
        for model_checkpoint in tqdm(model_checkpoints):
            score = load_and_evaluate_model(model_checkpoint, dataset)
            f.write(f"{util.get_info_from_model_file(model_checkpoint, 'name')}: {score}\n")
        

if __name__ == '__main__':
    root = ''
    modelsPath = 'lm_models/models/models_info.json'
    annFileTest = 'annotations/test_annotations_obj1.json'
    dataset_test = LineMODCocoDataset(root, annFileTest, modelsPath)
    
    # load_and_evaluate_models('model_checkpoints/', dataset_test, write_file='results.log')
    model_checkpoint = 'model_checkpoints/obj1_checkpoint_epochs60_lr0.0001_batch_size64_stages3_extra_convFalse.pth'
    model = DeepPose(
        extra_conv=util.get_info_from_model_file(model_checkpoint, 'extra_conv'),
        num_final_stages=util.get_info_from_model_file(model_checkpoint, 'stages')
    )
    load_model(model_checkpoint, model)
    score = evaluate(model=model,
                     dataset=dataset_test, 
                     obj_model=dataset_test.models['1'], 
                     model_name=util.get_info_from_model_file(model_checkpoint, 'name'), 
                     pic_samples=10)
    print('score is:', model_checkpoint, score)