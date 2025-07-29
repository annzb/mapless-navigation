import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def show_radar_clouds(clouds, prob_flags, titles=[],
                      intensity_threshold_percent=0.0,
                      window_name="Radar Visualization"):
    if len(clouds) != len(prob_flags):
        raise ValueError("clouds and prob_flags must have the same length.")
    n = len(clouds)

    # 1) global XYZ extents
    all_xyz = np.vstack([c[:, :3] for c in clouds])
    xyz_min = all_xyz.min(axis=0)
    xyz_max = all_xyz.max(axis=0)

    # 2) fixed 2 columns
    ncols = 2
    nrows = math.ceil(n / 2)

    # 3) make figure
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    fig.suptitle(window_name)
    axes = [
        fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        for i in range(n)
    ]
    cmap = plt.get_cmap("plasma")

    # 4) plot each
    for i, (ax, cloud, is_prob) in enumerate(zip(axes, clouds, prob_flags)):
        vals = cloud[:, 3]
        if is_prob:
            norm = np.clip(vals, 0.0, 1.0)
        else:
            lo, hi = vals.min(), vals.max()
            norm = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)

        if intensity_threshold_percent > 0:
            thr = np.percentile(norm, intensity_threshold_percent)
            mask = norm >= thr
        else:
            mask = np.ones_like(norm, bool)

        pts = cloud[mask, :3]
        colors = cmap(norm[mask])[:, :3]
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors, s=2, depthshade=True)

        # same axes limits & equal aspect
        ax.set_xlim(xyz_min[0], xyz_max[0])
        ax.set_ylim(xyz_min[1], xyz_max[1])
        ax.set_zlim(xyz_min[2], xyz_max[2])
        ax.set_box_aspect((1,1,1))

        if titles and i < len(titles):
            plot_title = titles[i]
        else:
            plot_title = "Prob" if is_prob else "Intensity"
        ax.set_title(plot_title)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # plt.title(window_name)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.show(block=True)


def transform_cloud(cloud, pose):
    pts = cloud[:, :3]
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    transformed = (pose @ pts_h.T).T[:, :3]
    new_cloud = np.copy(cloud)
    new_cloud[:, :3] = transformed
    return new_cloud


def get_map(lidar_clouds, poses, t):
    clouds = [
        transform_cloud(lidar_clouds[i], poses[i])
        for i in range(t + 1)
    ]
    return np.concatenate(clouds, axis=0)


def animate_map(lidar_clouds, radar_clouds, poses):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle("Use A/D keys to step backward/forward")

    current_step = [0]

    def update_plot(step):
        ax.cla()
        map_cloud = get_map(lidar_clouds, poses, step)
        radar_cloud = transform_cloud(radar_clouds[step], poses[step])

        m = ax.scatter(map_cloud[:, 0], map_cloud[:, 1], map_cloud[:, 2],
                       c=map_cloud[:, 3], cmap='gray', s=1, alpha=0.5)
        r = ax.scatter(radar_cloud[:, 0], radar_cloud[:, 1], radar_cloud[:, 2],
                       c=radar_cloud[:, 3], cmap='autumn', s=4)

        ax.set_title(f"Step {step}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        plt.draw()

    def on_key(event):
        if event.key == 'd' and current_step[0] < len(lidar_clouds) - 1:
            current_step[0] += 1
            update_plot(current_step[0])
        elif event.key == 'a' and current_step[0] > 0:
            current_step[0] -= 1
            update_plot(current_step[0])

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot(current_step[0])
    plt.show()
