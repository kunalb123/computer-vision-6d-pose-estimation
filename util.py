import numpy as np
import json
import matplotlib.pyplot as plt
import os
import cv2


def get_cam_matrix(file='lm/camera.json'):
    with open(file, 'r') as f:
        camera_params = json.load(f)

    cx = camera_params['cx']
    cy = camera_params['cy']
    fx = camera_params['fx']
    fy = camera_params['fy']

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K


def get_info_from_model_file(model_file, info):
    # file in format: 'maybesomefolder/obj1_checkpoint_epochs60_lr0.0001_batch_size64_stages5_extra_convFalse.pth'
    ret = None
    if info == 'name':
        start_index = model_file.find('_checkpoint')
        while model_file[start_index] != '/' and start_index > 0:
            start_index -= 1
        ret = model_file[start_index:model_file.find('.pth')]
    elif info == 'extra_conv':
        ret = model_file[model_file.find(info) + len(info):model_file.find('.pth')] == "True"
    elif info == 'lr':
        ret = float(model_file[model_file.find(info) + len(info):model_file.find('_batch_size')])
    else:
        start_index = model_file.find(info) + len(info)
        end_index = start_index
        ret = ''
        while model_file[end_index] != '_':
            ret += model_file[end_index]
            end_index += 1
        ret = int(ret)
    return ret

def plot_object_point_map(detected_object, projected_vertices, img, save_dir='eval_pics', model_name='model', name=''):
    if not os.path.exists(save_dir):
        print('creating directory', save_dir)
        os.makedirs(save_dir)
    if not os.path.exists(save_dir+'/'+model_name):
        print('creating directory', save_dir+'/'+model_name)
        os.makedirs(save_dir+'/'+model_name)
    
    od_points = np.array(detected_object['projected_points'])
    gt_points = np.array(projected_vertices)
    centroid_x = np.mean(gt_points[0, :])
    centroid_y = np.mean(gt_points[1, :])
    plt.clf()
    plt.imshow(img)
    plt.scatter(od_points[:,0], od_points[:,1], c='b', s=30)
    plt.scatter(gt_points[0, :], gt_points[1, :], c='r', s=20)
    plt.scatter([centroid_x], [centroid_y], c='g', s=20)
    plt.savefig(f'{save_dir}/{model_name}/{name}')


def visualize_vector_field():
    raise(NotImplementedError)
    batch_size = pred_vector_fields.shape[0]
    num_vertices = pred_vector_fields.shape[2]

    # Define the grid coordinates
    y, x = np.mgrid[0:pred_vector_fields.shape[3], 0:pred_vector_fields.shape[4]]

    # Plot each vector field in the batch
    for i in range(batch_size):
        for j in range(num_vertices // 2):  # Corrected indexing
            # Get the x and y components of the vector field for the current vertex
            vector_field_x = pred_vector_fields[i, 0, j*2].to(torch.device('mps')).numpy()
            vector_field_y = pred_vector_fields[i, 0, j*2+1].to(torch.device('mps')).numpy()

            plt.figure(figsize=(8, 6))
            plt.quiver(x, y, vector_field_x, vector_field_y, scale=30)
            plt.title(f'Vector Field {j+1} (Batch {i+1})')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')
            # plt.show()

def create_3D_vertices(model):
    vertices = np.empty((3, 0))
    x, y, z = model['min_x'], model['min_y'], model['min_z']
    size_x, size_y, size_z = np.array([model['size_x'], 0, 0]).reshape(3, -1),\
                                np.array([0, model['size_y'], 0]).reshape(3, -1),\
                                np.array([0, 0, model['size_z']]).reshape(3, -1)
    v1 = np.array([x, y, z]).reshape(3, -1)
    v2 = v1 + size_x
    v3 = v1 + size_y
    v4 = v1 + size_z
    v5 = v1 + size_x + size_y
    v6 = v1 + size_x + size_z
    v7 = v1 + size_y + size_z
    v8 = v1 + size_x + size_y + size_z
    vertices = np.concatenate([v1, v2, v3, v4, v5, v6, v7, v8], axis=1)
    #vertices = np.concatenate([eval(f'v{i}') for i in np.arange(1, 9)], axis=1)
    return vertices

def visualize_bbox(image, vertices, save_dir, model_name, name):
    image = np.require(image, requirements=['C_CONTIGUOUS'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = [
        (1, 4), (1, 5), (5, 7), (4, 7),  # Front face edges
        (0, 3), (0, 2), (2, 6), (3, 6),  # Back face edges
        (0, 1), (3, 5), (6, 7), (2, 4)   # Connecting edges between front and back faces
    ]
    overlay = image.copy()
    # Fill the faces to make the bounding box opaque
    front_face = np.array([vertices[1], vertices[4], vertices[7], vertices[5]])
    top_face = np.array([vertices[3], vertices[5], vertices[7], vertices[6]])
    back_face = np.array([vertices[0], vertices[2], vertices[6], vertices[3]])
    bottom_face = np.array([vertices[0], vertices[1], vertices[4], vertices[2]])
    right_face = np.array([vertices[2], vertices[4], vertices[7], vertices[6]])
    left_face = np.array([vertices[0], vertices[1], vertices[5], vertices[3]])
    alpha = 0.8 

    color = (209, 245, 66)


    cv2.fillConvexPoly(image, np.int32([front_face]), color)
    cv2.fillConvexPoly(image, np.int32([back_face]), color)
    cv2.fillConvexPoly(image, np.int32([top_face]), color)
    cv2.fillConvexPoly(image, np.int32([bottom_face]), color)
    cv2.fillConvexPoly(image, np.int32([right_face]), color)
    cv2.fillConvexPoly(image, np.int32([left_face]), color)
    # Draw the edges on the image
    
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    for edge in edges:
        pt1 = tuple(vertices[edge[0]].astype(np.uint32))
        pt2 = tuple(vertices[edge[1]].astype(np.uint32))
        cv2.line(image, pt1, pt2, color, 2)

    square_size = 3
    for vertex in vertices:
        top_left = (vertex[0] - square_size, vertex[1] - square_size)
        bottom_right = (vertex[0] + square_size, vertex[1] + square_size)
        cv2.rectangle(image, np.int32(top_left), np.int32(bottom_right), color, -1)  # Filled square with blue color

    # Show and save the image
    if not os.path.exists(save_dir):
        print('creating directory', save_dir)
        os.makedirs(save_dir)
    if not os.path.exists(save_dir+'/'+model_name):
        print('creating directory', save_dir+'/'+model_name)
        os.makedirs(save_dir+'/'+model_name)
    
    cv2.imwrite(f'{save_dir}/{model_name}/{name}.png', image)