import h5py
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import train_test_split

from utils.data_transforms import NumpyDataTransform, filter_point_intensity, polar_grid_to_cartesian_points, scale_point_intensity
from utils.radar_config import RadarConfig


def read_h5_dataset(file_path):
    print('Reading dataset from', file_path)
    data_dict = {}
    with h5py.File(file_path, 'r') as f:
        config = json.loads(f['config'][()])
        data_content = config.get('data_content', [])
        runs = config.get('runs', [])
        for content in data_content:
            data_dict[content] = {}
            for run in runs:
                dataset_name = f"{content}_{run}"
                sizes_dataset_name = f"{dataset_name}_sizes"
                if dataset_name in f:
                    if sizes_dataset_name in f:
                        flat_data = f[dataset_name][:]
                        sizes = f[sizes_dataset_name][:]
                        offsets = np.cumsum(sizes)
                        pointclouds = np.split(flat_data, offsets[:-1])
                        data_dict[content][run] = pointclouds
                    else:
                        data_dict[content][run] = f[dataset_name][:]
                else:
                    print(f"Dataset {dataset_name} not found in the file.")
    return data_dict, RadarConfig.from_dict(config.get('radar_config', {}))


# def process_radar_frames(radar_frames, intensity_mean=None, intensity_std=None):
#     radar_clouds = polar_grid_to_cartesian_points(radar_frames)
#     points_filtered, batch_indices_filtered = filter_point_intensity(radar_points)
#     radar_clouds, empty_cloud_idx = [], []
#     for radar_cloud in radar_clouds:
            
    
#     (points_scaled, batch_indices_scaled), intensity_mean = scale_point_intensity(
#         points=points_filtered, batch_indices=batch_indices_filtered, 
#         intensity_mean=intensity_mean, intensity_std=intensity_std
#     )
#     # points_filtered, batch_indices_filtered = filter_point_intensity(points=points_scaled[0], batch_indices=points_scaled[1])
#     return (points_filtered, batch_indices_filtered), intensity_mean, intensity_std


# def process_lidar_frames(lidar_frames, data_buffer=None):
#     empty_cloud_idx = []
#     for i in range(len(lidar_frames)):
#         if len(lidar_frames[i]) == 0:
#             empty_cloud_idx.append(i)
#         else:
#             lidar_frames[i][..., 3] = 1 / (1 + np.exp(-lidar_frames[i][..., 3]))
#     return lidar_frames, np.array(empty_cloud_idx)


def process_lidar_frames(lidar_frames, data_buffer=None):
    filtered_frames, nonempty_cloud_idx = [], []
    for i in range(len(lidar_frames)):
        if len(lidar_frames[i]) > 0:
            nonempty_cloud_idx.append(i)
            new_frame = lidar_frames[i].astype(np.float32, copy=True)
            new_frame[..., 3] = 1 / (1 + np.exp(-new_frame[..., 3]))
            filtered_frames.append(new_frame)
    return filtered_frames, np.array(nonempty_cloud_idx)


def prepare_point_data(
        radar_frames, lidar_frames, poses, data_transformer,
        dataset_part=1.0, logger=None
    ):
        assert len(radar_frames) == len(lidar_frames) == len(poses)
        assert 0.0 < dataset_part <= 1.0
        log_fn = logger.log if logger is not None else print

        Y, nonempty_lidar_idx = process_lidar_frames(lidar_frames)                                   # remove empty lidar clouds + convert log odds -> probs
        X = data_transformer.polar_grid_to_cartesian_points(grids=radar_frames[nonempty_lidar_idx])  # remove heatmaps with empty ground truth + convert to points
        X, nonempty_radar_idx = data_transformer.filter_point_intensity(points=X, threshold=0.09)   # remove points with 0 intensity from every radar cloud + remove empty clouds
        Y = [Y[i] for i in nonempty_radar_idx]                                                       # remove ground truth where input is empty
        poses = poses[nonempty_lidar_idx][nonempty_radar_idx]
        log_fn("Filtered", len(radar_frames) - len(X), "empty samples out of", len(radar_frames), ".")

        assert len(X) == len(Y) == len(poses) != 0
        if dataset_part < 1:
            target_num_samples = int(len(radar_frames) * dataset_part)
            X = X[:target_num_samples]
            Y = Y[:target_num_samples]
            poses = poses[:target_num_samples, ...]
        return X, Y, poses

class RadarDataset(Dataset):
    def __init__(
            self, radar_frames, lidar_frames, poses, data_transformer, *args,
            intensity_mean=None, intensity_std=None, data_buffer=None,
            name='dataset', logger=None, **kwargs
    ):
        self.X, self.intensity_mean, self.intensity_std = data_transformer.scale_point_intensity(
            points=radar_frames, intensity_mean=intensity_mean, intensity_std=intensity_std
        )
        self.Y = lidar_frames
        self.poses = poses
        self.name = name.capitalize()
        if logger is not None:
            self.print_log(logger=logger)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        radar_frame = self.X[index]
        lidar_frame = self.Y[index]
        pose = self.poses[index]
        return radar_frame, lidar_frame, pose

    @staticmethod
    def custom_collate_fn(batch):
        radar_list, lidar_list, poses = zip(*batch)
        poses = default_collate(poses)

        radar_batch, radar_batch_indices, lidar_batch, lidar_batch_indices = [], [], [], []
        for i, (radar_cloud, lidar_cloud, _) in enumerate(batch):
            radar_batch.append(torch.as_tensor(radar_cloud, dtype=torch.float32))
            lidar_batch.append(torch.as_tensor(lidar_cloud, dtype=torch.float32))
            radar_batch_indices.extend([i] * radar_cloud.shape[0])
            lidar_batch_indices.extend([i] * lidar_cloud.shape[0])

        radar_batch, lidar_batch = torch.cat(radar_batch, dim=0), torch.cat(lidar_batch, dim=0)
        radar_batch_indices, lidar_batch_indices = torch.as_tensor(radar_batch_indices, dtype=torch.int64), torch.as_tensor(lidar_batch_indices, dtype=torch.int64)
        return (radar_batch, radar_batch_indices), (lidar_batch, lidar_batch_indices), poses

    def print_log(self, logger=None):
        log_fn = logger.log if logger is not None else print
        X_counts = np.array([p.shape[0] for p in self.X])
        Y_counts = np.array([p.shape[0] for p in self.Y])
        log_fn(f"{self.name}: {len(self)} samples")
        log_fn(f"Radar (X) points per sample: "
              f"min={X_counts.min()}, max={X_counts.max()}, "
              f"mean={X_counts.mean():.1f}, median={np.median(X_counts)}")
        log_fn(f"Lidar (Y) points per sample: "
              f"min={Y_counts.min()}, max={Y_counts.max()}, "
              f"mean={Y_counts.mean():.1f}, median={np.median(Y_counts)}")


class RadarDatasetGrid(RadarDataset):
    def __init__(self, *args, radar_config, voxel_size=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        # self.X = clouds_to_grids(self.X, voxel_size, radar_config.point_range)
        # self.Y = clouds_to_grids(self.Y, radar_config)

    # @staticmethod
    # def custom_collate_fn(batch):
    #     radar_frames, lidar_frames, poses = zip(*batch)
    #     radar_frames = torch.tensor(radar_frames)
    #     lidar_frames = torch.tensor(lidar_frames)
    #     poses = torch.tensor(poses)
    #     return radar_frames, lidar_frames, poses

    def print_log(self):
        print(f'{self.name} input shape:', self.X.shape)
        # print(f'{self.name} output shape:', self.Y.shape)


def get_dataset(
        dataset_file_path, dataset_type,
        partial=1.0, batch_size=16, shuffle_runs=True, random_state=42, grid_voxel_size=1.0,
        data_buffer=None, device=None, logger=None
):
    data_dict, radar_config = read_h5_dataset(dataset_file_path)
    radar_frames = data_dict['cascade_heatmaps']
    lidar_frames = data_dict['lidar_map_samples']
    poses = data_dict['cascade_poses']

    if shuffle_runs:
        radar_frames = np.concatenate(list(radar_frames.values()), axis=0)
        lidar_frames = [np.array(frame) for run_frames in lidar_frames.values() for frame in run_frames]
        poses = np.concatenate(list(poses.values()), axis=0)
    else:
        raise NotImplementedError("Non-shuffled runs are not implemented.")
    # print('radar_frames.shape', radar_frames.shape)
    _, num_elevation_bins, num_azimuth_bins, num_range_bins = radar_frames.shape
    radar_config.set_radar_frame_params(num_azimuth_bins=num_azimuth_bins, num_range_bins=num_range_bins, num_elevation_bins=num_elevation_bins, grid_voxel_size=grid_voxel_size)
    # print('point range:', radar_config.point_range)

    # filter empty clouds
    # filtered_indices = [i for i, frame in enumerate(lidar_frames) if len(frame) > 0 and (any(frame[:, 3] >= occupancy_threshold) if occupied_only else True)]  # TODO: fix for grid
    # print(f'Filtered {len(radar_frames) - len(filtered_indices)} empty frames out of {len(radar_frames)}.')
    # radar_frames = np.array(radar_frames[filtered_indices])
    # lidar_frames = [lidar_frames[i] for i in filtered_indices]
    # poses = poses[filtered_indices]

    # reduce dataset
    # num_samples = int(len(radar_frames) * partial)
    # radar_frames = radar_frames[:num_samples]
    # lidar_frames = lidar_frames[:num_samples]
    # poses = poses[:num_samples]
    # print(num_samples, 'samples total.')

    data_transformer = NumpyDataTransform(radar_config)
    radar_frames_filtered, lidar_frames_filtered, poses_filtered = prepare_point_data(
        radar_frames, lidar_frames, poses,
        data_transformer=data_transformer, dataset_part=partial, logger=logger
    )

    # for i, sample in enumerate(lidar_frames):
    #     valid, resolution = validate_octomap_pointcloud(sample, tolerance=1e-2)
    #     if valid:
    #         if i == 0:
    #             print('valid sample', i, 'resolution:', resolution)
    #     else:
    #         print('invalid sample', i, 'resolution:', resolution)

    radar_train, radar_temp, lidar_train, lidar_temp, poses_train, poses_temp = train_test_split(radar_frames_filtered, lidar_frames_filtered, poses_filtered, test_size=0.5, random_state=random_state)
    radar_val, radar_test, lidar_val, lidar_test, poses_val, poses_test = train_test_split(radar_temp, lidar_temp, poses_temp, test_size=0.6, random_state=random_state)

    # dataset_class = RadarDatasetGrid if grid else RadarDataset
    train_dataset = dataset_type(radar_train, lidar_train, poses_train, data_transformer=data_transformer, data_buffer=data_buffer, device=device, radar_config=radar_config, voxel_size=grid_voxel_size, name='train')
    val_dataset = dataset_type(radar_val, lidar_val, poses_val, data_transformer=data_transformer, data_buffer=data_buffer, device=device, radar_config=radar_config, voxel_size=grid_voxel_size, intensity_mean=train_dataset.intensity_mean, intensity_std=train_dataset.intensity_std, name='valid')
    test_dataset = dataset_type(radar_test, lidar_test, poses_test, data_transformer=data_transformer, data_buffer=data_buffer, device=device, radar_config=radar_config, voxel_size=grid_voxel_size, intensity_mean=train_dataset.intensity_mean, intensity_std=train_dataset.intensity_std, name='test')

    # train_dataset.print_log()
    # val_dataset.print_log()
    # test_dataset.print_log()
    # print()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset_type.custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_type.custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_type.custom_collate_fn)

    return train_loader, val_loader, test_loader, radar_config
