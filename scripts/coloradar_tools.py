import os

import numpy as np
import struct


def read_tf_file(filename):
    if not os.path.exists(filename):
        print('File ' + filename + ' not found')
        return

    with open(filename, mode='r') as file:
        lines = file.readlines()

    t = [float(s) for s in lines[0].split()]
    r = [float(s) for s in lines[1].split()]

    return t, r


def read_waveform_cfg(wave_filename, wave_params):
    wave_params['data_type'] = 'adc_samples'

    with open(wave_filename, mode='r') as file:
        lines = file.readlines()

    int_vals = ['num_rx', 'num_tx', 'num_adc_samples_per_chirp', 'num_chirps_per_frame']
    for line in lines:
        vals = line.split()
        if vals[0] in int_vals:
            wave_params[vals[0]] = int(vals[1])
        else:
            wave_params[vals[0]] = float(vals[1])

    return wave_params


def read_heatmap_cfg(hm_filename, hm_params):
    hm_params['data_type'] = 'heatmap'

    with open(hm_filename, mode='r') as file:
        lines = file.readlines()

    for line in lines:
        vals = line.split()

        if vals[0] == 'azimuth_bins' or vals[0] == 'elevation_bins':
            hm_params[vals[0]] = [float(s) for s in vals[1:]]
        elif vals[0] == 'range_bin_width':
            hm_params[vals[0]] = float(vals[1])
        else:
            hm_params[vals[0]] = int(vals[1])

    return hm_params


def read_antenna_cfg(filename):
    antenna_cfg = {}

    with open(filename, mode='r') as file:
        lines = file.readlines()

    for line in lines:
        vals = line.split()

        if vals[0] != '#':
            if vals[0] == 'num_rx':
                antenna_cfg['num_rx'] = int(vals[1])
                antenna_cfg['rx_locations'] = [0] * antenna_cfg['num_rx']
            elif vals[0] == 'num_tx':
                antenna_cfg['num_tx'] = int(vals[1])
                antenna_cfg['tx_locations'] = [0] * antenna_cfg['num_tx']
            elif vals[0] == 'rx':
                antenna_cfg['rx_locations'][int(vals[1])] = (int(vals[2]), int(vals[3]))
            elif vals[0] == 'tx':
                antenna_cfg['tx_locations'][int(vals[1])] = (int(vals[2]), int(vals[3]))
            elif vals[0] == 'F_design':
                antenna_cfg['F_design'] = float(vals[1])

    return antenna_cfg


def read_coupling_cfg(coupling_filename):
    coupling_calib = {}

    with open(coupling_filename, mode='r') as file:
        lines = file.readlines()

    num_tx = int(lines[0].split(':')[1])
    num_rx = int(lines[1].split(':')[1])
    num_range_bins = int(lines[2].split(':')[1])
    coupling_calib['num_tx'] = num_tx
    coupling_calib['num_rx'] = num_rx
    coupling_calib['num_range_bins'] = num_range_bins
    coupling_calib['num_doppler_bins'] = int(lines[3].split(':')[1])
    data_str = lines[4].split(':')[1]
    data_arr = np.array(data_str.split(',')).astype('float')
    data_arr = data_arr[:-1:2] + 1j * data_arr[1::2]
    coupling_calib['data'] = data_arr.reshape(num_tx, num_rx, num_range_bins)

    return coupling_calib


def get_single_chip_params(calib_dir):
    wave_params = {'sensor_type': 'single_chip'}
    hm_params = {'sensor_type': 'single_chip'}
    pc_params = {'sensor_type': 'single_chip', 'data_type': 'pointcloud'}

    tf_filename = calib_dir + '/transforms/base_to_single_chip.txt'

    t, r = read_tf_file(tf_filename)

    wave_params['translation'] = t
    wave_params['rotation'] = r
    hm_params['translation'] = t
    hm_params['rotation'] = r
    pc_params['translation'] = t
    pc_params['rotation'] = r

    wave_filename = calib_dir + '/single_chip/waveform_cfg.txt'
    wave_params = read_waveform_cfg(wave_filename, wave_params)

    hm_filename = calib_dir + '/single_chip/heatmap_cfg.txt'
    hm_params = read_heatmap_cfg(hm_filename, hm_params)

    antenna_filename = calib_dir + '/single_chip/antenna_cfg.txt'
    antenna_cfg = read_antenna_cfg(antenna_filename)

    coupling_filename = calib_dir + '/single_chip/coupling_calib.txt'
    coupling_calib = read_coupling_cfg(coupling_filename)

    return {'waveform': wave_params,
            'heatmap': hm_params,
            'pointcloud': pc_params,
            'antenna': antenna_cfg,
            'coupling': coupling_calib}


def get_heatmap(filename, params):
    if not os.path.exists(filename):
        raise ValueError(f'File {filename} not found')
    with open(filename, mode='rb') as file:
        frame_bytes = file.read()

    frame_vals = struct.unpack(str(len(frame_bytes) // 4) + 'f', frame_bytes)
    frame_vals = np.array(frame_vals)
    frame = frame_vals.reshape((params['num_elevation_bins'],
                                params['num_azimuth_bins'],
                                params['num_range_bins'],
                                2))  # 2 vals for each bin (doppler peak intensity and peak location)
    return frame
