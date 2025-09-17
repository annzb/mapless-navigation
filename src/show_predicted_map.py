import os

from utils import evaluation as eval_utils
from utils.params import get_params
from train_points_generative import PointModelManager
from visualize import points as visualize


def run():
    model_path = '/Users/anna/data/rmodels/sweep1/29_best_train_loss.pth'
    save_file = 'predictions.pkl'

    params = get_params()
    params['device_name'] = 'cpu'
    params['dataset_params']['partial'] = 1.0
    # params['loss_params']['occupancy_threshold'] = 0.6

    mm = PointModelManager(**params)
    mm.init_model(model_path=model_path)

    if not os.path.exists(save_file):
        eval_utils.save_predictions(mm, save_file)
    radar_clouds, gt_clouds, predicted_clouds, poses, metrics = eval_utils.read_predictions(save_file)

    visualize.animate_map(radar_clouds, gt_clouds, poses)


if __name__ == '__main__':
    run()
