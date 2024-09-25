#include "coloradar_tools.h"


coloradar::SingleChipConfig::SingleChipConfig(const std::filesystem::path& calibDir) {
    coloradar::internal::checkPathExists(calibDir);
    init(calibDir);
}

void coloradar::SingleChipConfig::init(const std::filesystem::path& calibDir) {
    std::filesystem::path configDir = calibDir / "single_chip";
    coloradar::internal::checkPathExists(configDir);
    std::filesystem::path antennaConfigFilePath = configDir / "antenna_cfg.txt";
    std::filesystem::path heatmapConfigFilePath = configDir / "heatmap_cfg.txt";
    std::filesystem::path waveformConfigFilePath = configDir / "waveform_cfg.txt";
    std::filesystem::path couplingConfigFilePath = configDir / "coupling_calib.txt";
    initAntennaParams(antennaConfigFilePath);
    initHeatmapParams(heatmapConfigFilePath);
    initWaveformParams(waveformConfigFilePath);
    initCouplingParams(couplingConfigFilePath);
}

coloradar::CascadeConfig::CascadeConfig(const std::filesystem::path& calibDir) {
    coloradar::internal::checkPathExists(calibDir);
    init(calibDir);
}

void coloradar::CascadeConfig::init(const std::filesystem::path& calibDir) {
    std::filesystem::path configDir = calibDir / "cascade";
    coloradar::internal::checkPathExists(configDir);
    std::filesystem::path antennaConfigFilePath = configDir / "antenna_cfg.txt";
    std::filesystem::path heatmapConfigFilePath = configDir / "heatmap_cfg.txt";
    std::filesystem::path waveformConfigFilePath = configDir / "waveform_cfg.txt";
    std::filesystem::path couplingConfigFilePath = configDir / "coupling_calib.txt";
    std::filesystem::path phaseFrequencyConfigFilePath = configDir / "phase_frequency_calib.txt";
    initAntennaParams(antennaConfigFilePath);
    initHeatmapParams(heatmapConfigFilePath);
    initWaveformParams(waveformConfigFilePath);
    initCouplingParams(couplingConfigFilePath);
    initPhaseFrequencyParams(phaseFrequencyConfigFilePath);
}


void coloradar::RadarConfig::initAntennaParams(const std::filesystem::path& antennaCfgFile) {
    coloradar::internal::checkPathExists(antennaCfgFile);
    std::ifstream infile(antennaCfgFile);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open antenna config file: " + antennaCfgFile.string());
    }
    bool hasNumRxAntennas = false, hasNumTxAntennas = false, hasDesignFrequency = false;
    std::string key;
    txDistance.clear();
    txElevation.clear();
    rxDistance.clear();
    rxElevation.clear();

    while (infile >> key) {
        if (key == "num_rx") {
            infile >> numRxAntennas;
            hasNumRxAntennas = true;
        } else if (key == "num_tx") {
            infile >> numTxAntennas;
            hasNumTxAntennas = true;
        } else if (key == "F_design") {
            infile >> designFrequency;
            hasDesignFrequency = true;
        } else if (key == "rx") {
            int index;
            double distance, elevation;
            infile >> index >> distance >> elevation;
            rxDistance.push_back(distance);
            rxElevation.push_back(elevation);
        } else if (key == "tx") {
            int index;
            double distance, elevation;
            infile >> index >> distance >> elevation;
            txDistance.push_back(distance);
            txElevation.push_back(elevation);
        } else {
            throw std::runtime_error("Unknown key in antenna config file: " + key);
        }
    }
    infile.close();
    if (!hasNumRxAntennas || !hasNumTxAntennas || !hasDesignFrequency) {
        throw std::runtime_error("Missing required parameters in the antenna config file.");
    }
    if (rxDistance.size() != static_cast<size_t>(numRxAntennas)) {
        throw std::runtime_error("Mismatch in number of RX antennas. Expected: " + std::to_string(numRxAntennas) + ", Found: " + std::to_string(rxDistance.size()));
    }
    if (txDistance.size() != static_cast<size_t>(numTxAntennas)) {
        throw std::runtime_error("Mismatch in number of TX antennas. Expected: " + std::to_string(numTxAntennas) + ", Found: " + std::to_string(txDistance.size()));
    }
}


void coloradar::RadarConfig::initHeatmapParams(const std::filesystem::path& heatmapCfgFile) {
    coloradar::internal::checkPathExists(heatmapCfgFile);
    std::ifstream infile(heatmapCfgFile);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open heatmap config file: " + heatmapCfgFile.string());
    }
    bool hasNumRangeBins = false, hasNumElevationBins = false, hasNumAzimuthBins = false, hasRangeBinWidth = false, hasAzimuthBins = false, hasElevationBins = false;
    std::string key;

    while (infile >> key) {
        if (key == "num_range_bins") {
            infile >> numRangeBins;
            hasNumRangeBins = true;
        } else if (key == "num_elevation_bins") {
            infile >> numElevationBins;
            hasNumElevationBins = true;
        } else if (key == "num_azimuth_bins") {
            infile >> numAzimuthBins;
            hasNumAzimuthBins = true;
        } else if (key == "range_bin_width") {
            infile >> rangeBinWidth;
            hasRangeBinWidth = true;
        } else if (key == "azimuth_bins") {
            azimuthBins.clear();
            double binValue;
            while (infile.peek() != '\n' && infile >> binValue) {
                azimuthBins.push_back(binValue);
            }
            hasAzimuthBins = true;
        } else if (key == "elevation_bins") {
            elevationBins.clear();
            double binValue;
            while (infile.peek() != '\n' && infile >> binValue) {
                elevationBins.push_back(binValue);
            }
            hasElevationBins = true;
        } else {
            throw std::runtime_error("Unknown key in config file: " + key);
        }
    }
    infile.close();
    if (!hasNumRangeBins || !hasNumElevationBins || !hasNumAzimuthBins || !hasRangeBinWidth || !hasAzimuthBins || !hasElevationBins) {
        throw std::runtime_error("Missing required parameters in the heatmap config file.");
    }
    if (azimuthBins.size() != static_cast<size_t>(numAzimuthBins)) {
        throw std::runtime_error("Mismatch in number of azimuth bins. Expected: " + std::to_string(numAzimuthBins) + ", Found: " + std::to_string(azimuthBins.size()));
    }
    if (elevationBins.size() != static_cast<size_t>(numElevationBins)) {
        throw std::runtime_error("Mismatch in number of elevation bins. Expected: " + std::to_string(numElevationBins) + ", Found: " + std::to_string(elevationBins.size()));
    }
}

void coloradar::RadarConfig::initWaveformParams(const std::filesystem::path& waveformCfgFile) {
    coloradar::internal::checkPathExists(waveformCfgFile);
    std::ifstream infile(waveformCfgFile);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open waveform config file: " + waveformCfgFile.string());
    }
    bool hasNumAdcSamplesPerChirp = false, hasNumChirpsPerFrame = false, hasAdcSampleFrequency = false, hasStartFrequency = false, hasIdleTime = false, hasAdcStartTime = false, hasRampEndTime = false, hasFrequencySlope = false;
    int _;
    std::string key;

    while (infile >> key) {
        if (key == "num_rx") {
            infile >> _;
        } else if (key == "num_tx") {
            infile >> _;
        } else if (key == "num_adc_samples_per_chirp") {
            infile >> numAdcSamplesPerChirp;
            hasNumAdcSamplesPerChirp = true;
        } else if (key == "num_chirps_per_frame") {
            infile >> hasNumChirpsPerFrame;
            hasNumChirpsPerFrame = true;
        } else if (key == "adc_sample_frequency") {
            infile >> adcSampleFrequency;
            hasAdcSampleFrequency = true;
        } else if (key == "start_frequency") {
            infile >> startFrequency;
            hasStartFrequency = true;
        } else if (key == "idle_time") {
            infile >> idleTime;
            hasIdleTime = true;
        } else if (key == "adc_start_time") {
            infile >> adcStartTime;
            hasAdcStartTime = true;
        } else if (key == "ramp_end_time") {
            infile >> rampEndTime;
            hasRampEndTime = true;
        } else if (key == "frequency_slope") {
            infile >> frequencySlope;
            hasFrequencySlope = true;
        } else {
            throw std::runtime_error("Unknown key in waveform config file: " + key);
        }
    }
    infile.close();
    if (!hasNumAdcSamplesPerChirp || !hasNumChirpsPerFrame || !hasAdcSampleFrequency || !hasStartFrequency || !hasIdleTime || !hasAdcStartTime || !hasRampEndTime || !hasFrequencySlope) {
        throw std::runtime_error("Missing required parameters in the waveform config file.");
    }
}

void coloradar::RadarConfig::initCouplingParams(const std::filesystem::path& couplingCfgFile) {
    coloradar::internal::checkPathExists(couplingCfgFile);
    std::ifstream infile(couplingCfgFile);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open coupling calib config file: " + couplingCfgFile.string());
    }
    bool hasNumDopplerBins = false, hasCouplingCalibMatrix = false;
    int _;
    std::string key;
    couplingCalibMatrix.clear();

    while (infile >> key) {
        if (key == "num_rx") {
            infile >> _;
        } else if (key == "num_tx") {
            infile >> _;
        } else if (key == "num_range_bins") {
            infile >> _;
        } else if (key == "num_doppler_bins") {
            infile >> numDopplerBins;
            hasNumDopplerBins = true;
        } else if (key == "data") {
            std::string dataLine;
            std::getline(infile, dataLine);
            std::stringstream ss(dataLine);
            std::string valueStr;
            while (std::getline(ss, valueStr, ',')) {
                size_t pos;
                double real = std::stod(valueStr, &pos);
                double imag = std::stod(valueStr.substr(pos + 1));
                couplingCalibMatrix.push_back(std::complex<double>(real, imag));
            }
            size_t expectedSize = numTxAntennas * numRxAntennas * numRangeBins;
            if (couplingCalibMatrix.size() != expectedSize) {
                throw std::runtime_error("Mismatch in the size of the coupling calibration matrix. Expected: " + std::to_string(expectedSize) + ", Found: " + std::to_string(couplingCalibMatrix.size()));
            }
            hasCouplingCalibMatrix = true;
        } else {
            throw std::runtime_error("Unknown key in coupling calib config file: " + key);
        }
    }
    infile.close();
    if (!hasNumDopplerBins || !hasCouplingCalibMatrix) {
        throw std::runtime_error("Missing required parameters in the coupling calib config file.");
    }
}

void coloradar::RadarConfig::initPhaseFrequencyParams(const std::filesystem::path& phaseFrequencyCfgFile) {
    coloradar::internal::checkPathExists(phaseFrequencyCfgFile);
    std::ifstream infile(phaseFrequencyCfgFile);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open phase frequency calib config file: " + phaseFrequencyCfgFile.string());
    }
    frequencyCalibMatrix.clear();
    phaseCalibMatrix.clear();
    bool hasFrequencyCalibMatrix = false, hasPhaseCalibMatrix = false;

    std::string line;
    while (std::getline(infile, line)) {
        if (line.find("\"frequencyCalibrationMatrix\":") != std::string::npos) {
            size_t startPos = line.find('[') + 1;
            size_t endPos = line.find(']');
            std::string matrixStr = line.substr(startPos, endPos - startPos);
            std::stringstream ss(matrixStr);
            std::string valueStr;
            while (std::getline(ss, valueStr, ',')) {
                frequencyCalibMatrix.push_back(std::stod(valueStr));
            }
            size_t expectedSize = numTxAntennas * numRxAntennas;
            if (frequencyCalibMatrix.size() != expectedSize) {
                throw std::runtime_error("Mismatch in the size of the frequency calibration matrix. Expected: " + std::to_string(expectedSize) + ", Found: " + std::to_string(frequencyCalibMatrix.size()));
            }
            hasFrequencyCalibMatrix = true;
        } else if (line.find("\"phaseCalibrationMatrix\":") != std::string::npos) {
            size_t startPos = line.find('[') + 1;
            size_t endPos = line.find(']');
            std::string matrixStr = line.substr(startPos, endPos - startPos);
            std::stringstream ss(matrixStr);
            std::string valueStr;
            std::vector<double> phaseReal;
            std::vector<double> phaseImag;
            while (std::getline(ss, valueStr, ',')) {
                double real = std::stod(valueStr);
                std::getline(ss, valueStr, ',');
                double imag = std::stod(valueStr);
                phaseCalibMatrix.push_back(std::complex<double>(real, imag));
            }
            size_t expectedSize = numTxAntennas * numRxAntennas;
            if (phaseCalibMatrix.size() != expectedSize) {
                throw std::runtime_error("Mismatch in the size of the phase calibration matrix. Expected: " + std::to_string(expectedSize) + ", Found: " + std::to_string(phaseCalibMatrix.size()));
            }
            hasPhaseCalibMatrix = true;
        }
    }
    infile.close();
    if (!hasFrequencyCalibMatrix || !hasPhaseCalibMatrix) {
        throw std::runtime_error("Missing required calibration matrices in the phase frequency calib config file.");
    }
}
