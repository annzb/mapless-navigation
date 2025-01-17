from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import numpy as np


@dataclass
class RadarConfig:
    # Actual heatmap parameters
    num_range_bins: Optional[int]
    num_azimuth_bins: Optional[int]
    num_elevation_bins: Optional[int]
    clipped_azimuth_bins: Optional[List[float]]
    clipped_elevation_bins: Optional[List[float]]
    num_radar_points: Optional[int]
    point_range: Optional[Tuple]
    grid_size: Optional[Tuple]
    grid_resolution: Optional[float]
    num_grid_voxels: Optional[int]

    # Initial heatmap parameters
    total_range_bins: int
    total_elevation_bins: int
    total_azimuth_bins: int
    range_bin_width: float
    azimuth_bins: List[float]
    elevation_bins: List[float]

    # Antenna parameters
    design_frequency: float
    num_tx_antennas: int
    num_rx_antennas: int
    tx_centers: List[dict]
    rx_centers: List[dict]

    # Waveform parameters
    num_adc_samples_per_chirp: int
    num_chirps_per_frame: int
    adc_sample_frequency: float
    start_frequency: float
    idle_time: float
    adc_start_time: float
    ramp_end_time: float
    frequency_slope: float

    # Calibration parameters
    num_doppler_bins: int
    coupling_calib_matrix: List[complex]

    # Phase frequency parameters
    calib_adc_sample_frequency: float
    calib_frequency_slope: float
    frequency_calib_matrix: List[complex]
    phase_calib_matrix: List[complex]

    # Internal parameters
    num_azimuth_beams: int
    num_elevation_beams: int
    azimuth_aperture_len: float
    elevation_aperture_len: float
    num_angles: int
    num_virtual_elements: int
    virtual_array_map: List[int]
    azimuth_angles: List[float]
    elevation_angles: List[float]
    doppler_bin_width: float

    @classmethod
    def from_dict(cls, radar_config_dict: dict) -> "RadarConfig":
        return cls(
            total_range_bins=radar_config_dict["heatmap"]["numPosRangeBins"],
            total_elevation_bins=radar_config_dict["heatmap"]["numElevationBins"],
            total_azimuth_bins=radar_config_dict["heatmap"]["numAzimuthBins"],
            range_bin_width=radar_config_dict["heatmap"]["rangeBinWidth"],
            azimuth_bins=radar_config_dict["heatmap"]["azimuthBins"],
            elevation_bins=radar_config_dict["heatmap"]["elevationBins"],

            design_frequency=radar_config_dict["antenna"]["designFrequency"],
            num_tx_antennas=radar_config_dict["antenna"]["numTxAntennas"],
            num_rx_antennas=radar_config_dict["antenna"]["numRxAntennas"],
            tx_centers=radar_config_dict["antenna"]["txCenters"],
            rx_centers=radar_config_dict["antenna"]["rxCenters"],

            num_adc_samples_per_chirp=radar_config_dict["waveform"]["numAdcSamplesPerChirp"],
            num_chirps_per_frame=radar_config_dict["waveform"]["numChirpsPerFrame"],
            adc_sample_frequency=radar_config_dict["waveform"]["adcSampleFrequency"],
            start_frequency=radar_config_dict["waveform"]["startFrequency"],
            idle_time=radar_config_dict["waveform"]["idleTime"],
            adc_start_time=radar_config_dict["waveform"]["adcStartTime"],
            ramp_end_time=radar_config_dict["waveform"]["rampEndTime"],
            frequency_slope=radar_config_dict["waveform"]["frequencySlope"],

            num_doppler_bins=radar_config_dict["calibration"]["numDopplerBins"],
            coupling_calib_matrix=[
                complex(elem["real"], elem["imag"]) for elem in radar_config_dict["calibration"]["couplingCalibMatrix"]
            ],

            calib_adc_sample_frequency=radar_config_dict["phaseFrequency"]["calibAdcSampleFrequency"],
            calib_frequency_slope=radar_config_dict["phaseFrequency"]["calibFrequencySlope"],
            frequency_calib_matrix=[
                complex(elem["real"], elem["imag"]) for elem in
                radar_config_dict["phaseFrequency"]["frequencyCalibMatrix"]
            ],
            phase_calib_matrix=[
                complex(elem["real"], elem["imag"]) for elem in radar_config_dict["phaseFrequency"]["phaseCalibMatrix"]
            ],

            num_azimuth_beams=radar_config_dict["internal"]["numAzimuthBeams"],
            num_elevation_beams=radar_config_dict["internal"]["numElevationBeams"],
            azimuth_aperture_len=radar_config_dict["internal"]["azimuthApertureLen"],
            elevation_aperture_len=radar_config_dict["internal"]["elevationApertureLen"],
            num_angles=radar_config_dict["internal"]["numAngles"],
            num_virtual_elements=radar_config_dict["internal"]["numVirtualElements"],
            virtual_array_map=radar_config_dict["internal"]["virtualArrayMap"],
            azimuth_angles=radar_config_dict["internal"]["azimuthAngles"],
            elevation_angles=radar_config_dict["internal"]["elevationAngles"],
            doppler_bin_width=radar_config_dict["internal"]["dopplerBinWidth"],

            num_range_bins=None, num_azimuth_bins=None, num_elevation_bins=None,
            clipped_azimuth_bins=None, clipped_elevation_bins=None, num_radar_points=None,
            point_range=None, grid_size=None, grid_resolution=None, num_grid_voxels=None
        )

    def set_radar_frame_params(self, num_azimuth_bins: int, num_elevation_bins: int, num_range_bins: int, grid_voxel_size: float = 0.1):
        self.num_azimuth_bins = num_azimuth_bins
        self.num_elevation_bins = num_elevation_bins
        self.num_range_bins = num_range_bins
        azimuth_bins_to_clip = (self.total_azimuth_bins - num_azimuth_bins) // 2
        self.clipped_azimuth_bins = self.azimuth_bins[azimuth_bins_to_clip: self.total_azimuth_bins - azimuth_bins_to_clip]
        elevation_bins_to_clip = (self.total_elevation_bins - num_elevation_bins) // 2
        self.clipped_elevation_bins = self.elevation_bins[elevation_bins_to_clip: self.total_elevation_bins - elevation_bins_to_clip]

        self.num_radar_points = num_azimuth_bins * num_elevation_bins * num_range_bins
        y_max = self.num_range_bins * self.range_bin_width
        x_max = (y_max * np.tan(self.clipped_azimuth_bins[-1]) // grid_voxel_size) * grid_voxel_size
        z_max = (y_max * np.tan(self.clipped_elevation_bins[-1]) // grid_voxel_size) * grid_voxel_size
        self.point_range = (-x_max, x_max, 0, y_max, -z_max, z_max)
        self.grid_resolution = grid_voxel_size
        self.grid_size = (
            math.ceil(x_max * 2 / grid_voxel_size),
            math.ceil(y_max / grid_voxel_size),
            math.ceil(z_max * 2 / grid_voxel_size)
        )
        self.num_grid_voxels = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
