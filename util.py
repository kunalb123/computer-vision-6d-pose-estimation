import numpy as np
import json
import matplotlib.pyplot as plt
import os


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
        ret = bool(model_file[model_file.find(info) + len(info):model_file.find('.pth')])
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
    print('extracted info:', type(ret), ret)
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


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    Parameters:
    - quaternion: A list or numpy array with 4 elements [w, x, y, z]
    Returns:
    - A 3x3 rotation matrix
    """
    w, x, y, z = quaternion
    R = np.array([
    [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
    [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
    [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R


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