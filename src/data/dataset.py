import h5py
import json
import numpy as np
import psutil
import gc
import sys
from collections.abc import Mapping, Iterable
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import train_test_split

from data.data_transforms import NumpyDataTransform
from data.radar_config import RadarConfig


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


def process_lidar_frames(lidar_frames, occupancy_threshold=0.5, gt_cloud_min_num_points=1):
    nonempty_mask = np.array([len(frame) > gt_cloud_min_num_points for frame in lidar_frames])
    nonempty_cloud_idx = np.where(nonempty_mask)[0]
    if len(nonempty_cloud_idx) == 0:
        return [], np.array([], dtype=np.int64)
    
    has_occupied_mask = np.zeros(len(nonempty_cloud_idx), dtype=bool)
    for i, frame_idx in enumerate(nonempty_cloud_idx):
        lidar_frames[frame_idx][..., 3] = 1 / (1 + np.exp(-lidar_frames[frame_idx][..., 3]))
        occupied_cloud = lidar_frames[frame_idx][lidar_frames[frame_idx][..., 3] >= occupancy_threshold]
        has_occupied_mask[i] = len(occupied_cloud) > 0
    has_occupied_cloud_idx = nonempty_cloud_idx[has_occupied_mask]

    return [lidar_frames[i] for i in has_occupied_cloud_idx], has_occupied_cloud_idx


def get_size_recursive(obj):
    seen = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (str, bytes, int, float, bool, type(None))):
            return size
        elif isinstance(obj, Mapping):
            return size + sum(inner(k) + inner(v) for k, v in obj.items())
        elif isinstance(obj, Iterable):
            if hasattr(obj, '__len__') and len(obj) == 0:
                return size
            if isinstance(obj, list) and obj and isinstance(obj[0], np.ndarray):
                return size + sum(x.nbytes for x in obj)
            return size + sum(inner(x) for x in obj)
        return size
    return inner(obj)


def log_memory(stage, arrays=None, log_fn=print):
    log_fn(f"\n==== {stage.upper()} ====")
    process = psutil.Process()
    rss = process.memory_info().rss / (1024**3)
    log_fn(f"TOTAL MEMORY: {rss:.2f} GB")

    total_gb = 0
    if arrays:
        seen_ids = set()
        def get_size(obj):
            obj_id = id(obj)
            if obj_id in seen_ids:
                return 0
            seen_ids.add(obj_id)

            if isinstance(obj, np.ndarray):
                return obj.nbytes
            if isinstance(obj, (str, bytes, int, float, bool, type(None))):
                return sys.getsizeof(obj)
            if isinstance(obj, Mapping):
                size = sys.getsizeof(obj)
                for k, v in obj.items():
                    size += get_size(k) + get_size(v)
                return size
            if isinstance(obj, Iterable):
                size = sys.getsizeof(obj)
                if isinstance(obj, list) and obj and isinstance(obj[0], np.ndarray):
                    return size + sum(x.nbytes for x in obj)
                for item in obj:
                    size += get_size(item)
                return size
            return sys.getsizeof(obj)

        total_bytes = 0
        for name, arr in arrays.items():
            sz = get_size(arr)
            gb = sz / (1024**3)
            log_fn(f"    - {name}: {gb:.2f} GB")
            total_bytes += sz
        total_gb = total_bytes / (1024**3)
        log_fn(f"  TOTAL ARRAY MEMORY: {total_gb:.2f} GB")

    log_fn(f"  UNTRACKED MEMORY: {rss - total_gb:.2f} GB")


def prepare_grid_data(X, Y, poses, data_transformer, dataset_part=1.0, logger=None, **kwargs):
    """
    Remove samples with empty GT, convert lidar clouds to grids, convert lidar occupancy to probability. Convert radar polar grids to cartesian grids.
    X: polar grids
    Y: cartesian clouds
    poses: poses
    """
    assert len(X) == len(Y) == len(poses)
    assert 0.0 < dataset_part <= 1.0
    log_fn = logger.log if logger is not None else print
    orig_size = len(X)
    target_num_samples = int(orig_size * dataset_part)
    log_fn(f"Collecting {target_num_samples} samples out of {orig_size}.")
    log_memory("start", {'X': X, 'Y': Y, 'poses': poses})

    X_processed, Y_processed, poses_processed = [], [], []
    for i, (X_sample, Y_sample, pose_sample) in tqdm(enumerate(zip(X, Y, poses))):
        nonempty_Y_sample = Y_sample[Y_sample[..., 3] > 0]
        if len(nonempty_Y_sample) == 0:  # empty GT
            continue
        
        Y_sample_grid = data_transformer.point_clouds_to_cartesian_grid(nonempty_Y_sample)
        Y_sample_grid = 1 / (1 + np.exp(-Y_sample_grid))
        X_sample_cartesian = data_transformer.polar_grid_to_cartesian_grid(X_sample)

        X_processed.append(X_sample_cartesian)
        Y_processed.append(Y_sample_grid)
        poses_processed.append(pose_sample)
        if len(X_processed) >= target_num_samples:
            break

    X_processed, Y_processed, poses_processed = np.stack(X_processed), np.stack(Y_processed), np.stack(poses_processed)
    assert len(X_processed) == len(Y_processed) == len(poses_processed) != 0
    if dataset_part < 1:
        X_processed = X_processed[:target_num_samples]
        Y_processed = Y_processed[:target_num_samples]
        poses_processed = poses_processed[:target_num_samples]
    log_memory("sliced for dataset_part", {
        'X': X, 'Y': Y, 'poses': poses, 
        'X_processed': X_processed, 'Y_processed': Y_processed, 'poses_processed': poses_processed
    })
    return X_processed, Y_processed, poses_processed


def prepare_point_data(
        X, Y, poses, data_transformer, 
        dataset_part=1.0, occupancy_threshold=0.5, intensity_threshold=0.0, gt_cloud_min_num_points=1,
        logger=None, **kwargs
):
    assert len(X) == len(Y) == len(poses)
    assert 0.0 < dataset_part <= 1.0
    log_fn = logger.log if logger is not None else print
    orig_size = len(X)
    target_num_samples = int(orig_size * dataset_part)
    log_fn(f"Collecting {target_num_samples} samples out of {orig_size}.")
    log_memory("start", {'X': X, 'Y': Y, 'poses': poses})

    batch_size = 100
    n_iter = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
    X_processed, Y_processed, poses_processed = [], [], []
    for i in tqdm(range(n_iter)):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        Y_batch = Y[i * batch_size:(i + 1) * batch_size]
        poses_batch = poses[i * batch_size:(i + 1) * batch_size]
        
        Y_batch, nonempty_lidar_idx = process_lidar_frames(Y_batch, occupancy_threshold=occupancy_threshold, gt_cloud_min_num_points=gt_cloud_min_num_points)
        if len(Y_batch) == 0:
            continue

        X_batch = data_transformer.polar_grid_to_cartesian_points(X_batch[nonempty_lidar_idx])
        X_batch, nonempty_radar_idx = data_transformer.filter_point_intensity(points=X_batch, threshold=intensity_threshold)
        if len(X_batch) == 0:
            continue

        Y_batch = [Y_batch[i] for i in nonempty_radar_idx]
        poses_batch = poses_batch[nonempty_lidar_idx][nonempty_radar_idx]
        X_processed.extend(X_batch)
        Y_processed.extend(Y_batch)
        poses_processed.extend(poses_batch)
        if len(X_processed) >= target_num_samples:
            break

    assert len(X_processed) == len(Y_processed) == len(poses_processed) != 0
    if dataset_part < 1:
        X_processed = X_processed[:target_num_samples]
        Y_processed = Y_processed[:target_num_samples]
        poses_processed = poses_processed[:target_num_samples]
    log_memory("sliced for dataset_part", {
        'X': X, 'Y': Y, 'poses': poses, 
        'X_processed': X_processed, 'Y_processed': Y_processed, 'poses_processed': poses_processed
    })
        
    return X_processed, Y_processed, poses_processed


class RadarDataset(Dataset):
    def __init__(
            self, radar_frames, lidar_frames, poses, data_transformer,
            intensity_mean=None, intensity_std=None, coord_means=None, coord_stds=None,
            name='dataset', logger=None, orig_radar_frames=None, **kwargs
    ):
        self.X, self.coord_means, self.coord_stds = data_transformer.scale_point_coords(
            points=radar_frames, coord_means=coord_means, coord_stds=coord_stds
        )
        self.X, self.intensity_mean, self.intensity_std = data_transformer.scale_point_intensity(
            points=self.X, intensity_mean=intensity_mean, intensity_std=intensity_std
        )
        self.Y, _, _ = data_transformer.scale_point_coords(
            points=lidar_frames, coord_means=coord_means, coord_stds=coord_stds
        )
        self.poses = poses
        self.orig_radar_frames = orig_radar_frames
        self.name = name.capitalize()
        self.print_log(logger=logger)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        radar_frame = self.X[index]
        lidar_frame = self.Y[index]
        pose = self.poses[index]
        return radar_frame, lidar_frame, pose
    
    @staticmethod
    def preprocess(*args, **kwargs):
        return prepare_point_data(*args, **kwargs)

    @staticmethod
    def custom_collate_fn(batch):
        _, _, poses = zip(*batch)
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


class GridRadarDataset(RadarDataset):
    def __init__(
            self, 
            radar_frames,  # polar grids
            lidar_frames,  # cartesian clouds
            poses, data_transformer,
            intensity_mean=None, intensity_std=None,
            name='dataset', logger=None, **kwargs
    ):
        self.X, self.intensity_mean, self.intensity_std = data_transformer.scale_grid_intensity(
            grids=radar_frames, intensity_mean=intensity_mean, intensity_std=intensity_std
        )
        self.X = np.expand_dims(self.X.transpose(0, 3, 2, 1), axis=1)
        self.Y = np.expand_dims(lidar_frames.transpose(0, 3, 2, 1), axis=1)
        self.poses = poses
        self.name = name.capitalize()
        self.print_log(logger=logger)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        radar_frame = self.X[index]
        lidar_frame = self.Y[index]
        pose = self.poses[index]
        return radar_frame, lidar_frame, pose
    
    @staticmethod
    def preprocess(*args, **kwargs):
        return prepare_grid_data(*args, **kwargs)

    @staticmethod
    def custom_collate_fn(batch):
        return default_collate(batch)

    def print_log(self, logger=None):
        log_fn = logger.log if logger is not None else print
        log_fn(f"Radar (X) Shape: {self.X.shape}")
        log_fn(f"Lidar (Y) Shape: {self.Y.shape}")


def get_dataset(
        dataset_file_path, dataset_type,
        device=None, logger=None, random_seed=1, batch_size=1,
        partial=1.0, shuffle_runs=True, grid_voxel_size=1.0,
        gt_cloud_min_num_points=1, intensity_threshold=0.0, occupancy_threshold=0.5, **kwargs
):
    if not isinstance(dataset_file_path, str):
        raise ValueError('dataset_file_path must be a string')
    if not issubclass(dataset_type, Dataset):
        raise ValueError('dataset_type must be a Dataset object')
    if not isinstance(device, torch.device):
        raise ValueError('device must be a torch.device')
    if not isinstance(random_seed, int):
        raise ValueError('random_seed must be an integer')
    if not isinstance(batch_size, int):
        raise ValueError('batch_size must be an integer')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if not isinstance(partial, (int, float)):
        raise ValueError('partial must be a number')
    if partial <= 0 or partial > 1:
        raise ValueError('partial must be between 0 and 1')
    if not isinstance(shuffle_runs, bool):
        raise ValueError('shuffle_runs must be a boolean')
    if not isinstance(grid_voxel_size, (int, float)):
        raise ValueError('grid_voxel_size must be a number')
    if grid_voxel_size <= 0:
        raise ValueError('grid_voxel_size must be positive')
    if not isinstance(intensity_threshold, (int, float)):
        raise ValueError('intensity_threshold must be a number')
    if intensity_threshold < 0:
        raise ValueError('intensity_threshold must be non-negative')

    data_dict, radar_config = read_h5_dataset(dataset_file_path)
    radar_frames = data_dict['cascade_heatmaps']
    lidar_frames = data_dict['lidar_map_samples']
    poses = data_dict['cascade_poses']
    del data_dict
    gc.collect()

    if shuffle_runs:
        radar_frames = np.concatenate(list(radar_frames.values()), axis=0)
        lidar_frames = [np.array(frame) for run_frames in lidar_frames.values() for frame in run_frames]
        poses = np.concatenate(list(poses.values()), axis=0)
    else:
        raise NotImplementedError("Non-shuffled runs are not implemented.")
    _, num_elevation_bins, num_azimuth_bins, num_range_bins = radar_frames.shape
    radar_config.set_radar_frame_params(num_azimuth_bins=num_azimuth_bins, num_range_bins=num_range_bins, num_elevation_bins=num_elevation_bins, grid_voxel_size=grid_voxel_size)

    print(f"Preparing point data...")
    data_transformer = NumpyDataTransform(radar_config)
    radar_frames, lidar_frames, poses = dataset_type.preprocess(
        radar_frames, lidar_frames, poses,
        data_transformer=data_transformer, dataset_part=partial, logger=logger,
        intensity_threshold=intensity_threshold, occupancy_threshold=occupancy_threshold, gt_cloud_min_num_points=gt_cloud_min_num_points
    )
    (
        radar_train, radar_temp,
        lidar_train, lidar_temp,
        poses_train, poses_temp,
        # orig_radar_frames_train, orig_radar_frames_temp
    ) = train_test_split(
        radar_frames, lidar_frames, poses, # orig_radar_frames, 
        test_size=0.5, random_state=random_seed
    )
    (
        radar_val, radar_test,
        lidar_val, lidar_test,
        poses_val, poses_test,
        # orig_radar_frames_val, orig_radar_frames_test
    ) = train_test_split(
        radar_temp, lidar_temp, poses_temp, # orig_radar_frames_temp, 
        test_size=0.6, random_state=random_seed
    )
    print(f"Finished splitting dataset.")
    # dataset_class = RadarDatasetGrid if grid else RadarDataset
    train_dataset = dataset_type(
        radar_train, lidar_train, poses_train, 
        data_transformer=data_transformer, device=device, name='train',
        # orig_radar_frames=orig_radar_frames_train, 
        voxel_size=grid_voxel_size
    )
    val_dataset = dataset_type(
        radar_val, lidar_val, poses_val, 
        data_transformer=data_transformer, device=device, name='valid', 
        intensity_mean=train_dataset.intensity_mean, intensity_std=train_dataset.intensity_std, coord_means=train_dataset.coord_means, coord_stds=train_dataset.coord_stds,
        # orig_radar_frames=orig_radar_frames_val, 
        voxel_size=grid_voxel_size
    )
    test_dataset = dataset_type(
        radar_test, lidar_test, poses_test, 
        data_transformer=data_transformer, device=device, name='test', 
        intensity_mean=train_dataset.intensity_mean, intensity_std=train_dataset.intensity_std, coord_means=train_dataset.coord_means, coord_stds=train_dataset.coord_stds,
        # orig_radar_frames=orig_radar_frames_test, 
        voxel_size=grid_voxel_size
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset_type.custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_type.custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_type.custom_collate_fn)

    return train_loader, val_loader, test_loader, radar_config
