import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
import numpy as np
import json
from torch.utils.data import DataLoader
import cv2
import albumentations as A
import util

class LineMODCocoDataset(CocoDetection):
    def __init__(self, root, annFile, modelsPath, transform=None, target_transform=None):
        super(LineMODCocoDataset, self).__init__(root, annFile, transform, target_transform)
        self.models = json.load(open(modelsPath, 'r'))

    def project_3D_vertices(self, vertices, target):
        pose = self.extract_pose(target) # 3 x 4
        P_m2c = np.array(target['cam_K']).reshape(3, 3) @ pose
        vertices = np.vstack((vertices, np.ones((1, vertices.shape[1]))))
        homogeneous_2d = P_m2c @ vertices
        pixel_coordinates = homogeneous_2d[:2, :] / homogeneous_2d[2, :]
        return pixel_coordinates
    
    def gaussian_heatmaps(self, center, sigma, size):
        # Define the center and sigma
        center_x, center_y = center[0], center[1]

        # Create a grid of (x, y) coordinates
        y = np.linspace(0, size[0] - 1, size[0])
        x = np.linspace(0, size[1] - 1, size[1])
        x, y = np.meshgrid(x, y)

        # Compute the Gaussian function
        gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        return gaussian
    
    def vector_field(self, vertex, centroid, size):
        grid_height = size[0]
        grid_width = size[1]

        # Create a grid of (x, y) coordinates
        x = np.arange(grid_width)
        y = np.arange(grid_height)
        x, y = np.meshgrid(x, y)

        # Initialize the vector field with zeros (using float type)
        vector_field_x = np.zeros((grid_height, grid_width), dtype=float)
        vector_field_y = np.zeros((grid_height, grid_width), dtype=float)

        # Compute the distance from each pixel to the vertex
        distance = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2)

        # Set the radius within which the vector components will be computed
        radius = 3

        # Find the indices of pixels within the specified radius
        within_radius = distance <= radius

        # Compute the vector components pointing toward the centroid
        vector_x = centroid[0] - x.astype(float)
        vector_y = centroid[1] - y.astype(float)

        # Normalize the vectors
        magnitude = np.sqrt(vector_x**2 + vector_y**2)
        non_zero = magnitude > 0
        vector_x[non_zero] /= magnitude[non_zero]
        vector_y[non_zero] /= magnitude[non_zero]

        # Set the vector components within the radius
        vector_field_x[within_radius] = vector_x[within_radius]
        vector_field_y[within_radius] = vector_y[within_radius]
        return vector_field_x, vector_field_y

    def generate_ground_truth(self, image, target):
        h, w = image.shape[1] // 8, image.shape[2] // 8
        belief_map = np.zeros((9, h, w), dtype=np.float32)
        vector_field = np.zeros((16, h, w), dtype=np.float32)
        cat = target['category_id']
        model = self.models[str(cat)]
        vertices = util.create_3D_vertices(model)
        projected_vertices = self.project_3D_vertices(vertices, target)

        # Implement the actual logic for generating belief maps and vector fields
        # using 2D Gaussians and normalized vectors as described in the provided info.

        centroid = np.mean(projected_vertices, axis=1)

        for i, (px, py) in enumerate(projected_vertices.T):
            belief_map[i] = self.gaussian_heatmaps((px//8, py//8), 2, (h, w))
            vector_field[2*i], vector_field[2*i+1] = self.vector_field((px//8, py//8), centroid//8, (h, w))
        
        belief_map[8] = self.gaussian_heatmaps(centroid//8, 2, (h, w))

        return belief_map, vector_field, projected_vertices

    def __getitem__(self, index):
        out = super(LineMODCocoDataset, self).__getitem__(index)
        img, target = out[0], out[1][0]
        img = np.array(img).transpose((1, 2, 0))
        transform = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=1
                ),
                A.GaussNoise(p=1)
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        img = transform(image=img)['image'].transpose(2, 0, 1)
        img_norm = normalize(img)
        belief_map, vector_field, projected_vertices = self.generate_ground_truth(img_norm, target)
        gt_maps = np.concatenate((belief_map, vector_field), axis=0)
        gt_maps = torch.from_numpy(gt_maps).float()

        return img_norm, {'gt_maps': gt_maps, 'projected_vertices': projected_vertices, 'targets': target,'img': img}

    def extract_pose(self, target):
        # Placeholder for extracting pose information from the target
        # This should be implemented according to the specifics of your dataset
        R, t = np.array(target['cam_R_m2c']), np.array(target['cam_t_m2c'])
        pose = np.hstack([R.reshape(3, 3), t.reshape(3, -1)])
        return pose
