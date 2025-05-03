import h5py
import json
import numpy as np
import psutil
import gc
import sys
from collections.abc import Mapping, Iterable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import train_test_split

from utils.data_transforms import NumpyDataTransform
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


def process_lidar_frames(lidar_frames):
    nonempty_mask = np.array([len(frame) > 0 for frame in lidar_frames])
    nonempty_cloud_idx = np.where(nonempty_mask)[0]
    if len(nonempty_cloud_idx) == 0:
        return [], np.array([])
    
    for i in nonempty_cloud_idx:
        lidar_frames[i][..., 3] = 1 / (1 + np.exp(-lidar_frames[i][..., 3]))
    return [lidar_frames[i] for i in nonempty_cloud_idx], nonempty_cloud_idx


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
    log_fn(f'==== {stage.upper()} ====')
    process = psutil.Process()
    memory_info = process.memory_info()
    mem_usage = memory_info.rss / (1024 ** 3)  # GB
    log_fn(f"TOTAL MEMORY: {mem_usage:.2f} GB")
    
    total_size = 0
    if arrays:
        for name, arr in arrays.items():
            if isinstance(arr, np.ndarray):
                size_gb = arr.nbytes / (1024 ** 3)
                log_fn(f"    - {name}: {size_gb:.2f} GB, shape: {arr.shape}, dtype: {arr.dtype}")
                total_size += size_gb
            elif isinstance(arr, list) and arr and isinstance(arr[0], np.ndarray):
                total_bytes = sum(x.nbytes for x in arr)
                size_gb = total_bytes / (1024 ** 3)
                avg_points = sum(x.shape[0] for x in arr) / len(arr) if arr else 0
                log_fn(f"    - {name}: {size_gb:.2f} GB, avg {avg_points:.1f} points")
                total_size += size_gb
            else:
                size_bytes = get_size_recursive(arr)
                size_gb = size_bytes / (1024 ** 3)
                log_fn(f"    - {name}: {size_gb:.4f} GB (estimated recursively)")
                total_size += size_gb

        log_fn(f"  TOTAL ARRAY MEMORY: {total_size:.2f} GB")
    log_fn(f"  UNTRACKED MEMORY: {mem_usage - total_size:.2f} GB\n")


def prepare_point_data(X, Y, poses, data_transformer, dataset_part=1.0, logger=None):
        assert len(X) == len(Y) == len(poses)
        assert 0.0 < dataset_part <= 1.0
        log_fn = logger.log if logger is not None else print
        orig_size = len(X)
        log_memory("start", {'X': X, 'Y': Y, 'poses': poses})
        
        Y, nonempty_lidar_idx = process_lidar_frames(Y)
        log_memory("processed lidar frames", {'X': X, 'Y': Y, 'poses': poses, 'nonempty_lidar_idx': nonempty_lidar_idx})

        X = data_transformer.polar_grid_to_cartesian_points(X[nonempty_lidar_idx])
        log_memory("cartesian points", {'X': X, 'Y': Y, 'poses': poses, 'nonempty_lidar_idx': nonempty_lidar_idx})

        X, nonempty_radar_idx = data_transformer.filter_point_intensity(points=X, threshold=0.09)
        log_memory("filtered points", {'X': X, 'Y': Y, 'poses': poses, 'nonempty_lidar_idx': nonempty_lidar_idx, 'nonempty_radar_idx': nonempty_radar_idx})
        
        Y = [Y[i] for i in nonempty_radar_idx]
        log_memory("filtered Y", {'X': X, 'Y': Y, 'poses': poses, 'nonempty_lidar_idx': nonempty_lidar_idx, 'nonempty_radar_idx': nonempty_radar_idx})
        
        poses = poses[nonempty_lidar_idx][nonempty_radar_idx]
        log_fn("Filtered", orig_size - len(X), "empty samples out of", orig_size, ".")
        log_memory("filtered poses", {'X': X, 'Y': Y, 'poses': poses, 'nonempty_lidar_idx': nonempty_lidar_idx, 'nonempty_radar_idx': nonempty_radar_idx})

        assert len(X) == len(Y) == len(poses) != 0
        if dataset_part < 1:
            target_num_samples = int(orig_size * dataset_part)
            X = X[:target_num_samples]
            Y = Y[:target_num_samples]
            poses = poses[:target_num_samples, ...]
            log_memory("sliced for dataset_part", {'X': X, 'Y': Y, 'poses': poses})
            
        return X, Y, poses


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


def get_dataset(
        dataset_file_path, dataset_type,
        partial=1.0, batch_size=16, shuffle_runs=True, random_state=42, grid_voxel_size=1.0,
       device=None, logger=None, **kwargs
):
    log_memory("reading dataset")
    data_dict, radar_config = read_h5_dataset(dataset_file_path)
    radar_frames = data_dict['cascade_heatmaps']
    lidar_frames = data_dict['lidar_map_samples']
    poses = data_dict['cascade_poses']
    log_memory("dataset read", {'data_dict': data_dict, 'radar_frames': radar_frames, 'lidar_frames': lidar_frames, 'poses': poses})
    del data_dict
    gc.collect()
    log_memory("dataset collected", {'radar_frames': radar_frames, 'lidar_frames': lidar_frames, 'poses': poses})
    
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
    radar_frames, lidar_frames, poses = prepare_point_data(
        radar_frames, lidar_frames, poses,
        data_transformer=data_transformer, dataset_part=partial, logger=logger
    )
    (
        radar_train, radar_temp,
        lidar_train, lidar_temp,
        poses_train, poses_temp,
        # orig_radar_frames_train, orig_radar_frames_temp
    ) = train_test_split(
        radar_frames, lidar_frames, poses, # orig_radar_frames, 
        test_size=0.5, random_state=random_state
    )
    (
        radar_val, radar_test,
        lidar_val, lidar_test,
        poses_val, poses_test,
        # orig_radar_frames_val, orig_radar_frames_test
    ) = train_test_split(
        radar_temp, lidar_temp, poses_temp, # orig_radar_frames_temp, 
        test_size=0.6, random_state=random_state
    )

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
