from dataclasses import dataclass, field
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

    # Scaling parameters
    is_normalized: bool = False
    coord_means: Optional[np.ndarray] = field(default=None, repr=False)
    coord_stds: Optional[np.ndarray] = field(default=None, repr=False)
    scaled_point_range: Optional[Tuple] = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, radar_config_dict: dict) -> "RadarConfig":
        return cls(
            total_range_bins=radar_config_dict["heatmap"]["numRangeBins"],
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

    def scale_grid_parameters(self, coord_means: np.ndarray, coord_stds: np.ndarray):
        """
        Updates the grid bounds (point_range, grid_size) to match the normalized coordinate system.
        """
        if self.is_normalized:
            print("Warning: Grid parameters have already been scaled.")
            return
        if self.point_range is None:
            raise ValueError("Cannot scale grid parameters before they are initialized with set_radar_frame_params.")

        self.coord_means = coord_means
        self.coord_stds = coord_stds
        x_min, x_max, y_min, y_max, z_min, z_max = self.point_range
        scaled_x_min = (x_min - self.coord_means[0]) / self.coord_stds[0]
        scaled_x_max = (x_max - self.coord_means[0]) / self.coord_stds[0]
        scaled_y_min = (y_min - self.coord_means[1]) / self.coord_stds[1]
        scaled_y_max = (y_max - self.coord_means[1]) / self.coord_stds[1]
        scaled_z_min = (z_min - self.coord_means[2]) / self.coord_stds[2]
        scaled_z_max = (z_max - self.coord_means[2]) / self.coord_stds[2]
        self.scaled_point_range = (scaled_x_min, scaled_x_max, scaled_y_min, scaled_y_max, scaled_z_min, scaled_z_max)
        self.is_normalized = True
