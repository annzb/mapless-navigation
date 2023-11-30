import pickle

import numpy as np
from PIL import Image


def save_heatmap_image(heatmap, filename='heatmap.png', channels=(0, 1)):
    if len(channels) == 1 and channels[0] in (0, 1):
        heatmap_cropped = heatmap[:, :, :, channels[0]]
        rgb_image = np.zeros((heatmap_cropped.shape[1], heatmap_cropped.shape[2], 3), dtype=np.uint8)

        for x in range(heatmap_cropped.shape[1]):
            for y in range(heatmap_cropped.shape[2]):
                # Maximum intensity for green channel
                max_intensity = np.max(heatmap_cropped[:, x, y])

                # Slice index (elevation) for red channel
                # Assuming higher index corresponds to higher elevation
                slice_index = np.argmax(heatmap_cropped[:, x, y])

                # Normalize and set the RGB channels
                # Red channel: normalized intensity
                # Blue channel: normalized slice index
                # Green channel: constant (can be adjusted)
                rgb_image[x, y, 0] = int((max_intensity / np.max(heatmap_cropped)) * 255)
                rgb_image[x, y, 2] = int((slice_index / heatmap_cropped.shape[0]) * 255)
                rgb_image[x, y, 1] = 64  # Constant value for blue channel

        # Create and save the image
        img = Image.fromarray(rgb_image)
        img.save(filename)
    else:
        raise ValueError


if __name__ == "__main__":
    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    heatmaps = np.array(data['heatmaps'])
    for i in (0, 50, 100, 150, 200):
        save_heatmap_image(heatmaps[i], filename=f'heatmap_intensity_{i}.png', channels=(0, ))
