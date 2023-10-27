import os
import numpy as np

import rosbag
from sensor_msgs import point_cloud2

from .coloradar_tools import get_heatmap, get_single_chip_params


def main():
    coloradar_dir = '/home/ann/mapping/coloradar'
    run_folder_name = 'hallways_run0'
    calib_folder_name = 'calib'
    bag_file_path = os.path.join(coloradar_dir, 'ec_hallways_run0_lidar_octomap.bag')
    topic_name = "/lidar_filtered/octomap_point_cloud_centers"

    last_msg = None
    with rosbag.Bag(bag_file_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic_name]):
            last_msg = msg
    if last_msg:
        gen = point_cloud2.read_points(last_msg, field_names=("x", "y", "z"), skip_nans=True)
        true_map = np.array(list(gen))

    run_dir = os.path.join(coloradar_dir, run_folder_name)
    params = get_single_chip_params(calib_dir=os.path.join(coloradar_dir, calib_folder_name))

    num_files = len(os.listdir(os.path.join(run_dir, 'single_chip/heatmaps/data')))
    for i in range(num_files):
        heatmap = get_heatmap(
            filename=os.path.join(run_dir, 'single_chip/heatmaps/data/heatmap_' + str(i) + '.bin'),
            params=params['heatmap']
        )


if __name__ == '__main__':
    main()
