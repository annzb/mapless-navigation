import os
import pickle


subset_size = 100

if __name__ == '__main__':
    colab_root, local_root, brute_root = '/content/drive/My Drive', '/home/ann/mapping/mn_ws/src/mapless-navigation', '/home/annz/mapping/mn_ws/src/mapless-navigation'
    if os.path.isdir(colab_root):
        root = colab_root
    elif os.path.isdir(local_root):
        root = local_root
    else:
        root = brute_root
    dataset_filename = 'dataset_7runs_rangelimit.pkl'
    dataset_filepath = f'/media/giantdrive/coloradar/{dataset_filename}' if root == brute_root else os.path.join(root, dataset_filename)
    
    with open(dataset_filepath, 'rb') as f:
        data = pickle.load(f)
    print('Runs in dataset:', ', '.join(data.keys()))

    params = data.pop('params')
    gt_key = 'polar_grids'
    subset = {'params': params}
    for run_name in data:
        subset[run_name] = {
            'heatmaps': data[run_name]['heatmaps'][:subset_size],
            gt_key: data[run_name][gt_key][:subset_size],
            'poses': data[run_name]['poses'][:subset_size]
        }
        
    subset_file_name = dataset_filepath.replace('.pkl', f'_subset{subset_size}_polar.pkl')
    with open(subset_file_name, 'wb') as f:
        pickle.dump(subset, f)
