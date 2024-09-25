#include "coloradar_tools.h"

#include <map>
#include <sstream>
#include <regex>


std::map<std::string, std::vector<std::string>> readConfig(const std::filesystem::path& configFile) {
    std::cout << "Reading config " << configFile << std::endl;
    coloradar::internal::checkPathExists(configFile);
    std::ifstream infile(configFile);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open config file: " + configFile.string());
    }
    std::map<std::string, std::vector<std::string>> configMap;
    std::string line;
    std::regex keyValueRegex(R"(\s*([^#\s]+)\s*[:=\s]\s*(.*))");  // Match key and values

    while (std::getline(infile, line)) {
        // Skip empty lines and comments
        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos || line.find('#') == 0) {
            continue;
        }
        std::smatch match;
        if (std::regex_match(line, match, keyValueRegex)) {
            std::string key = match[1];    // Extract key
            std::string valueSection = match[2];  // Extract value section (everything after the key)

            std::istringstream valuesStream(valueSection);
            std::string value;
            std::vector<std::string> values;

            // Read all values, splitting by space
            while (valuesStream >> value) {
                values.push_back(value);
            }
            std::cout << "Processing key: " << key << " with " << values.size() << " values." << std::endl;
            configMap[key] = values;  // Store key-value pair in the map
        }
    }
    infile.close();
    return configMap;
}


void coloradar::RadarConfig::initAntennaParams(const std::filesystem::path& antennaCfgFile) {
    auto configMap = readConfig(antennaCfgFile);
    txDistance.clear();
    txElevation.clear();
    rxDistance.clear();
    rxElevation.clear();

    std::cout << "Processing config map " << antennaCfgFile << std::endl;
    for (const auto& [key, values] : configMap) {
        std::cout << key << ": ";
        for (const auto& value : values) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    auto it = configMap.find("num_rx");
    if (it != configMap.end()) {
        std::cout << "it->second[0] " << it->second[0] << std::endl;
        numRxAntennas = std::stoi(it->second[0]);
    } else {
        throw std::runtime_error("Missing num_rx in antenna config.");
    }
    std::cout << "num_rx " << numRxAntennas << std::endl;
    it = configMap.find("num_tx");
    if (it != configMap.end()) {
        numTxAntennas = std::stoi(it->second[0]);
    } else {
        throw std::runtime_error("Missing num_tx in antenna config.");
    }
    it = configMap.find("F_design");
    if (it != configMap.end()) {
        designFrequency = std::stod(it->second[0]);
    } else {
        throw std::runtime_error("Missing F_design in antenna config.");
    }
    std::cout << "F_design " << designFrequency << std::endl;
    for (size_t i = 0; i < numRxAntennas; ++i) {
        std::string key = "rx_" + std::to_string(i);
        it = configMap.find(key);
        if (it != configMap.end()) {
            rxDistance.push_back(std::stod(it->second[0]));
            rxElevation.push_back(std::stod(it->second[1]));
        }
    }
    for (size_t i = 0; i < numTxAntennas; ++i) {
        std::string key = "tx_" + std::to_string(i);
        it = configMap.find(key);
        if (it != configMap.end()) {
            txDistance.push_back(std::stod(it->second[0]));
            txElevation.push_back(std::stod(it->second[1]));
        }
    }
}

void coloradar::RadarConfig::initHeatmapParams(const std::filesystem::path& heatmapCfgFile) {
    auto configMap = readConfig(heatmapCfgFile);
    azimuthBins.clear();
    elevationBins.clear();

    auto it = configMap.find("num_range_bins");
    if (it != configMap.end()) {
        numRangeBins = std::stoi(it->second[0]);
    } else {
        throw std::runtime_error("Missing num_range_bins in heatmap config.");
    }
    it = configMap.find("num_elevation_bins");
    if (it != configMap.end()) {
        numElevationBins = std::stoi(it->second[0]);
    }
    it = configMap.find("num_azimuth_bins");
    if (it != configMap.end()) {
        numAzimuthBins = std::stoi(it->second[0]);
    }
    it = configMap.find("range_bin_width");
    if (it != configMap.end()) {
        rangeBinWidth = std::stod(it->second[0]);
    }
    it = configMap.find("azimuth_bins");
    if (it != configMap.end()) {
        for (const auto& bin : it->second) {
            azimuthBins.push_back(std::stod(bin));
        }
    }
    it = configMap.find("elevation_bins");
    if (it != configMap.end()) {
        for (const auto& bin : it->second) {
            elevationBins.push_back(std::stod(bin));
        }
    }
}


void coloradar::RadarConfig::initWaveformParams(const std::filesystem::path& waveformCfgFile) {
    auto configMap = readConfig(waveformCfgFile);

    auto it = configMap.find("num_adc_samples_per_chirp");
    if (it != configMap.end()) {
        numAdcSamplesPerChirp = std::stoi(it->second[0]);
    } else {
        throw std::runtime_error("Missing num_adc_samples_per_chirp in waveform config.");
    }
    it = configMap.find("num_chirps_per_frame");
    if (it != configMap.end()) {
        numChirpsPerFrame = std::stoi(it->second[0]);
    }
    it = configMap.find("adc_sample_frequency");
    if (it != configMap.end()) {
        adcSampleFrequency = std::stod(it->second[0]);
    }
    it = configMap.find("start_frequency");
    if (it != configMap.end()) {
        startFrequency = std::stod(it->second[0]);
    }
    it = configMap.find("idle_time");
    if (it != configMap.end()) {
        idleTime = std::stod(it->second[0]);
    }
    it = configMap.find("adc_start_time");
    if (it != configMap.end()) {
        adcStartTime = std::stod(it->second[0]);
    }
    it = configMap.find("ramp_end_time");
    if (it != configMap.end()) {
        rampEndTime = std::stod(it->second[0]);
    }
    it = configMap.find("frequency_slope");
    if (it != configMap.end()) {
        frequencySlope = std::stod(it->second[0]);
    }
}

void coloradar::RadarConfig::initCouplingParams(const std::filesystem::path& couplingCfgFile) {
    std::cout << "Start initCouplingParams" << std::endl;
    auto configMap = readConfig(couplingCfgFile);
    couplingCalibMatrix.clear();

    auto it = configMap.find("num_doppler_bins");
    if (it != configMap.end()) {
        std::cout << "it->second[0] " << it->second[0] << std::endl;
        numDopplerBins = std::stoi(it->second[0]);
    } else {
        throw std::runtime_error("Missing num_doppler_bins in coupling config.");
    }
    std::cout << "Read numDopplerBins " << numDopplerBins << std::endl;
    it = configMap.find("data");
    if (it != configMap.end()) {
        for (const auto& entry : it->second) {
            size_t pos;
            double real = std::stod(entry, &pos);
            double imag = std::stod(entry.substr(pos + 1));
            couplingCalibMatrix.push_back(std::complex<double>(real, imag));
        }
    } else {
        throw std::runtime_error("Missing coupling calibration data in config.");
    }
}

void coloradar::RadarConfig::initPhaseFrequencyParams(const std::filesystem::path& phaseFrequencyCfgFile) {
    std::cout << "Start initPhaseFrequencyParams" << std::endl;
    auto configMap = readConfig(phaseFrequencyCfgFile);
    frequencyCalibMatrix.clear();
    phaseCalibMatrix.clear();

    auto it = configMap.find("frequencyCalibrationMatrix");
    if (it != configMap.end()) {
        for (const auto& value : it->second) {
            frequencyCalibMatrix.push_back(std::stod(value));
        }
    } else {
        throw std::runtime_error("Missing frequencyCalibrationMatrix in phase frequency config.");
    }
    it = configMap.find("phaseCalibrationMatrix");
    if (it != configMap.end()) {
        for (size_t i = 0; i < it->second.size(); i += 2) {
            double real = std::stod(it->second[i]);
            double imag = std::stod(it->second[i + 1]);
            phaseCalibMatrix.push_back(std::complex<double>(real, imag));
        }
    } else {
        throw std::runtime_error("Missing phaseCalibrationMatrix in phase frequency config.");
    }
    std::cout << "Finish initPhaseFrequencyParams" << std::endl;
}


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
    std::cout << "Start CascadeConfig::init" << std::endl;
    std::filesystem::path configDir = calibDir / "cascade";
    coloradar::internal::checkPathExists(configDir);
    std::filesystem::path antennaConfigFilePath = configDir / "antenna_cfg.txt";
    std::filesystem::path heatmapConfigFilePath = configDir / "heatmap_cfg.txt";
    std::filesystem::path waveformConfigFilePath = configDir / "waveform_cfg.txt";
    std::filesystem::path couplingConfigFilePath = configDir / "coupling_calib.txt";
    std::filesystem::path phaseFrequencyConfigFilePath = configDir / "phase_frequency_calib.txt";
    std::cout << "Constructed paths" << std::endl;
    initAntennaParams(antennaConfigFilePath);
    std::cout << "Finish CascadeConfig::initAntennaParams" << std::endl;
    initHeatmapParams(heatmapConfigFilePath);
    std::cout << "Finish CascadeConfig::initHeatmapParams" << std::endl;
    initWaveformParams(waveformConfigFilePath);
    std::cout << "Finish CascadeConfig::initWaveformParams" << std::endl;
    initCouplingParams(couplingConfigFilePath);
    std::cout << "Finish CascadeConfig::initCouplingParams" << std::endl;
    initPhaseFrequencyParams(phaseFrequencyConfigFilePath);
    std::cout << "Finish CascadeConfig::init" << std::endl;
}


//void coloradar::RadarConfig::initAntennaParams(const std::filesystem::path& antennaCfgFile) {
//    coloradar::internal::checkPathExists(antennaCfgFile);
//    std::ifstream infile(antennaCfgFile);
//    if (!infile.is_open()) {
//        throw std::runtime_error("Unable to open antenna config file: " + antennaCfgFile.string());
//    }
//    bool hasNumRxAntennas = false, hasNumTxAntennas = false, hasDesignFrequency = false;
//    txDistance.clear();
//    txElevation.clear();
//    rxDistance.clear();
//    rxElevation.clear();
//    std::string line;
//
//    while (std::getline(infile, line)) {
//        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos || line.find('#') == 0) {
//            continue;
//        }
//        std::istringstream iss(line);
//        std::string key;
//        iss >> key;
//        if (key == "num_rx") {
//            iss >> numRxAntennas;
//            hasNumRxAntennas = true;
//        } else if (key == "num_tx") {
//            iss >> numTxAntennas;
//            hasNumTxAntennas = true;
//        } else if (key == "F_design") {
//            iss >> designFrequency;
//            hasDesignFrequency = true;
//        } else if (key == "rx") {
//            int index;
//            double distance, elevation;
//            iss >> index >> distance >> elevation;
//            rxDistance.push_back(distance);
//            rxElevation.push_back(elevation);
//        } else if (key == "tx") {
//            int index;
//            double distance, elevation;
//            iss >> index >> distance >> elevation;
//            txDistance.push_back(distance);
//            txElevation.push_back(elevation);
//        } else {
//            throw std::runtime_error("Unknown key in antenna config file: " + key);
//        }
//    }
//    infile.close();
//    if (!hasNumRxAntennas || !hasNumTxAntennas || !hasDesignFrequency) {
//        throw std::runtime_error("Missing required parameters in the antenna config file.");
//    }
//    if (rxDistance.size() != static_cast<size_t>(numRxAntennas)) {
//        throw std::runtime_error("Mismatch in number of RX antennas. Expected: " + std::to_string(numRxAntennas) + ", Found: " + std::to_string(rxDistance.size()));
//    }
//    if (txDistance.size() != static_cast<size_t>(numTxAntennas)) {
//        throw std::runtime_error("Mismatch in number of TX antennas. Expected: " + std::to_string(numTxAntennas) + ", Found: " + std::to_string(txDistance.size()));
//    }
//}
//
//void coloradar::RadarConfig::initHeatmapParams(const std::filesystem::path& heatmapCfgFile) {
//    coloradar::internal::checkPathExists(heatmapCfgFile);
//    std::ifstream infile(heatmapCfgFile);
//    if (!infile.is_open()) {
//        throw std::runtime_error("Unable to open heatmap config file: " + heatmapCfgFile.string());
//    }
//    bool hasNumRangeBins = false, hasNumElevationBins = false, hasNumAzimuthBins = false, hasRangeBinWidth = false, hasAzimuthBins = false, hasElevationBins = false;
//    std::string line;
//
//    while (std::getline(infile, line)) {
//        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos || line.find('#') == 0) {
//            continue;
//        }
//        std::istringstream iss(line);
//        std::string key;
//        iss >> key;
//        if (key == "num_range_bins") {
//            iss >> numRangeBins;
//            hasNumRangeBins = true;
//        } else if (key == "num_elevation_bins") {
//            iss >> numElevationBins;
//            hasNumElevationBins = true;
//        } else if (key == "num_azimuth_bins") {
//            iss >> numAzimuthBins;
//            hasNumAzimuthBins = true;
//        } else if (key == "range_bin_width") {
//            iss >> rangeBinWidth;
//            hasRangeBinWidth = true;
//        } else if (key == "azimuth_bins") {
//            azimuthBins.clear();
//            double binValue;
//            while (iss.peek() != '\n' && iss >> binValue) {
//                azimuthBins.push_back(binValue);
//            }
//            hasAzimuthBins = true;
//        } else if (key == "elevation_bins") {
//            elevationBins.clear();
//            double binValue;
//            while (iss.peek() != '\n' && iss >> binValue) {
//                elevationBins.push_back(binValue);
//            }
//            hasElevationBins = true;
//        } else {
//            throw std::runtime_error("Unknown key in config file: " + key);
//        }
//    }
//    infile.close();
//    if (!hasNumRangeBins || !hasNumElevationBins || !hasNumAzimuthBins || !hasRangeBinWidth || !hasAzimuthBins || !hasElevationBins) {
//        throw std::runtime_error("Missing required parameters in the heatmap config file.");
//    }
//    if (azimuthBins.size() != static_cast<size_t>(numAzimuthBins)) {
//        throw std::runtime_error("Mismatch in number of azimuth bins. Expected: " + std::to_string(numAzimuthBins) + ", Found: " + std::to_string(azimuthBins.size()));
//    }
//    if (elevationBins.size() != static_cast<size_t>(numElevationBins)) {
//        throw std::runtime_error("Mismatch in number of elevation bins. Expected: " + std::to_string(numElevationBins) + ", Found: " + std::to_string(elevationBins.size()));
//    }
//}
//
//void coloradar::RadarConfig::initWaveformParams(const std::filesystem::path& waveformCfgFile) {
//    coloradar::internal::checkPathExists(waveformCfgFile);
//    std::ifstream infile(waveformCfgFile);
//    if (!infile.is_open()) {
//        throw std::runtime_error("Unable to open waveform config file: " + waveformCfgFile.string());
//    }
//    bool hasNumAdcSamplesPerChirp = false, hasNumChirpsPerFrame = false, hasAdcSampleFrequency = false, hasStartFrequency = false, hasIdleTime = false, hasAdcStartTime = false, hasRampEndTime = false, hasFrequencySlope = false;
//    std::string line;
//
//    while (std::getline(infile, line)) {
//        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos || line.find('#') == 0) {
//            continue;
//        }
//        std::istringstream iss(line);
//        std::string key;
//        iss >> key;
//        if (key == "num_rx") {
//            int _;
//            iss >> _;
//        } else if (key == "num_tx") {
//            int _;
//            iss >> _;
//        } else if (key == "num_adc_samples_per_chirp") {
//            iss >> numAdcSamplesPerChirp;
//            hasNumAdcSamplesPerChirp = true;
//        } else if (key == "num_chirps_per_frame") {
//            iss >> numChirpsPerFrame;
//            hasNumChirpsPerFrame = true;
//        } else if (key == "adc_sample_frequency") {
//            iss >> adcSampleFrequency;
//            hasAdcSampleFrequency = true;
//        } else if (key == "start_frequency") {
//            iss >> startFrequency;
//            hasStartFrequency = true;
//        } else if (key == "idle_time") {
//            iss >> idleTime;
//            hasIdleTime = true;
//        } else if (key == "adc_start_time") {
//            iss >> adcStartTime;
//            hasAdcStartTime = true;
//        } else if (key == "ramp_end_time") {
//            iss >> rampEndTime;
//            hasRampEndTime = true;
//        } else if (key == "frequency_slope") {
//            iss >> frequencySlope;
//            hasFrequencySlope = true;
//        } else {
//            throw std::runtime_error("Unknown key in waveform config file: " + key);
//        }
//    }
//    infile.close();
//    if (!hasNumAdcSamplesPerChirp || !hasNumChirpsPerFrame || !hasAdcSampleFrequency || !hasStartFrequency || !hasIdleTime || !hasAdcStartTime || !hasRampEndTime || !hasFrequencySlope) {
//        throw std::runtime_error("Missing required parameters in the waveform config file.");
//    }
//}
//
//void coloradar::RadarConfig::initCouplingParams(const std::filesystem::path& couplingCfgFile) {
//    coloradar::internal::checkPathExists(couplingCfgFile);
//    std::ifstream infile(couplingCfgFile);
//    if (!infile.is_open()) {
//        throw std::runtime_error("Unable to open coupling calib config file: " + couplingCfgFile.string());
//    }
//    bool hasNumDopplerBins = false, hasCouplingCalibMatrix = false;
//    std::string line;
//    couplingCalibMatrix.clear();
//
//    while (std::getline(infile, line)) {
//        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos || line.find('#') == 0) {
//            continue;
//        }
//        std::istringstream iss(line);
//        std::string key;
//        iss >> key;
//        if (key == "num_rx") {
//            int _;
//            iss >> _;
//        } else if (key == "num_tx") {
//            int _;
//            iss >> _;
//        } else if (key == "num_range_bins") {
//            int _;
//            iss >> _;
//        } else if (key == "num_doppler_bins") {
//            iss >> numDopplerBins;
//            hasNumDopplerBins = true;
//        } else if (key == "data") {
//            std::string dataLine;
//            std::getline(infile, dataLine);
//            std::stringstream ss(dataLine);
//            std::string valueStr;
//            while (std::getline(ss, valueStr, ',')) {
//                size_t pos;
//                double real = std::stod(valueStr, &pos);
//                double imag = std::stod(valueStr.substr(pos + 1));
//                couplingCalibMatrix.push_back(std::complex<double>(real, imag));
//            }
//            size_t expectedSize = numTxAntennas * numRxAntennas * numRangeBins;
//            if (couplingCalibMatrix.size() != expectedSize) {
//                throw std::runtime_error("Mismatch in the size of the coupling calibration matrix. Expected: " + std::to_string(expectedSize) + ", Found: " + std::to_string(couplingCalibMatrix.size()));
//            }
//            hasCouplingCalibMatrix = true;
//        } else {
//            throw std::runtime_error("Unknown key in coupling calib config file: " + key);
//        }
//    }
//    infile.close();
//    if (!hasNumDopplerBins || !hasCouplingCalibMatrix) {
//        throw std::runtime_error("Missing required parameters in the coupling calib config file.");
//    }
//}
//
//void coloradar::RadarConfig::initPhaseFrequencyParams(const std::filesystem::path& phaseFrequencyCfgFile) {
//    coloradar::internal::checkPathExists(phaseFrequencyCfgFile);
//    std::ifstream infile(phaseFrequencyCfgFile);
//    if (!infile.is_open()) {
//        throw std::runtime_error("Unable to open phase frequency calib config file: " + phaseFrequencyCfgFile.string());
//    }
//    frequencyCalibMatrix.clear();
//    phaseCalibMatrix.clear();
//    bool hasFrequencyCalibMatrix = false, hasPhaseCalibMatrix = false;
//
//    std::string line;
//    while (std::getline(infile, line)) {
//        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos || line.find('#') == 0) {
//            continue;
//        }
//        if (line.find("\"frequencyCalibrationMatrix\":") != std::string::npos) {
//            size_t startPos = line.find('[') + 1;
//            size_t endPos = line.find(']');
//            std::string matrixStr = line.substr(startPos, endPos - startPos);
//            std::stringstream ss(matrixStr);
//            std::string valueStr;
//            while (std::getline(ss, valueStr, ',')) {
//                frequencyCalibMatrix.push_back(std::stod(valueStr));
//            }
//            size_t expectedSize = numTxAntennas * numRxAntennas;
//            if (frequencyCalibMatrix.size() != expectedSize) {
//                throw std::runtime_error("Mismatch in the size of the frequency calibration matrix. Expected: " + std::to_string(expectedSize) + ", Found: " + std::to_string(frequencyCalibMatrix.size()));
//            }
//            hasFrequencyCalibMatrix = true;
//        } else if (line.find("\"phaseCalibrationMatrix\":") != std::string::npos) {
//            size_t startPos = line.find('[') + 1;
//            size_t endPos = line.find(']');
//            std::string matrixStr = line.substr(startPos, endPos - startPos);
//            std::stringstream ss(matrixStr);
//            std::string valueStr;
//            while (std::getline(ss, valueStr, ',')) {
//                double real = std::stod(valueStr);
//                std::getline(ss, valueStr, ',');
//                double imag = std::stod(valueStr);
//                phaseCalibMatrix.push_back(std::complex<double>(real, imag));
//            }
//            size_t expectedSize = numTxAntennas * numRxAntennas;
//            if (phaseCalibMatrix.size() != expectedSize) {
//                throw std::runtime_error("Mismatch in the size of the phase calibration matrix. Expected: " + std::to_string(expectedSize) + ", Found: " + std::to_string(phaseCalibMatrix.size()));
//            }
//            hasPhaseCalibMatrix = true;
//        }
//    }
//    infile.close();
//    if (!hasFrequencyCalibMatrix || !hasPhaseCalibMatrix) {
//        throw std::runtime_error("Missing required calibration matrices in the phase frequency calib config file.");
//    }
//}
