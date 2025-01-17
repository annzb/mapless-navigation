from train_polar import run


if __name__ == '__main__':
    run(use_grid_data=True, octomap_voxel_size=0.25, model_save_name="best_grid_model.pth")
