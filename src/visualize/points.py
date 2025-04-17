import numpy as np
import matplotlib.pyplot as plt


def show_radar_clouds(clouds, prob_flags, intensity_threshold_percent=0.0, window_name="Radar Visualization"):
    """
    Display multiple radar point clouds in a single Matplotlib 3D window, coloring by intensity or probability.

    Args:
        clouds (list of np.ndarray): Each array is (N,4) with columns [x,y,z,intensity_or_prob].
        prob_flags (list of bool): Same length as clouds; True if the corresponding cloud's 4th column represents probabilities in [0,1], False if raw intensities.
        intensity_threshold_percent (float): Percentile threshold (0-100) to filter out low-intensity points.
        window_name (str): Title of the plot window.
    """
    if len(clouds) != len(prob_flags):
        raise ValueError("clouds and prob_flags must have the same length.")

    fig = plt.figure(window_name, figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap("plasma")

    for cloud, is_prob in zip(clouds, prob_flags):
        if cloud.ndim != 2 or cloud.shape[1] != 4:
            raise ValueError("Each cloud must be an (N,4) array.")
        intensities = cloud[:, 3]
        if is_prob:
            norm_vals = np.clip(intensities, 0.0, 1.0)
        else:
            lo, hi = intensities.min(), intensities.max()
            if hi > lo:
                norm_vals = (intensities - lo) / (hi - lo)
            else:
                norm_vals = np.zeros_like(intensities)

        if intensity_threshold_percent > 0:
            thresh = np.percentile(norm_vals, intensity_threshold_percent)
            mask = norm_vals >= thresh
        else:
            mask = np.ones_like(norm_vals, dtype=bool)
        pts = cloud[mask, :3]
        colors = cmap(norm_vals[mask])[:, :3]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=2, depthshade=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(window_name)
    plt.tight_layout()
    plt.show()
