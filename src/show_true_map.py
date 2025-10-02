from tqdm import tqdm

from utils.params import get_params
from train_points_generative import PointModelManager
from visualize import points as visualize


def run():
    params = get_params()
    params['device_name'] = 'cpu'
    params['dataset_params']['dataset_file_path'] = '/Users/anna/data/coloradar/coloradar_sep17_one.h5'
    params['dataset_params']['partial'] = 1.0
    params['dataset_params']['normalize_point_coords'] = False
    # params['loss_params']['occupancy_threshold'] = 0.6

    model_manager = PointModelManager(**params)

    radar_clouds, gt_clouds, poses = [], [], []
    for sample_idx in tqdm(range(len(model_manager.train_loader.dataset))):
        input_cloud_np, true_cloud_np, pose = model_manager.train_loader.dataset[sample_idx]
        radar_clouds.append(input_cloud_np)
        gt_clouds.append(true_cloud_np)
        poses.append(pose)

    visualize.animate_map(lidar_clouds=gt_clouds, radar_clouds=radar_clouds, poses=poses)


if __name__ == '__main__':
    run()
