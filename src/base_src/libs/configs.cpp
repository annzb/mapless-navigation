#include "coloradar_tools.h"

#include <json/json.h>
#include <map>
#include <sstream>
#include <regex>


std::map<std::string, std::vector<std::string>> readConfig(const std::filesystem::path& configFile) {
    coloradar::internal::checkPathExists(configFile);
    std::ifstream infile(configFile);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open config file: " + configFile.string());
    }
    std::map<std::string, std::vector<std::string>> configMap;
    std::string line;
    std::regex keyValueRegex(R"(\s*([^#\s]+)\s*[:=\s]\s*(.*))");
    const size_t maxKeySearchLength = 256;

    while (std::getline(infile, line)) {
        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos || line.find('#') == 0) {
            continue;
        }
        std::string lineHead = line.substr(0, maxKeySearchLength);
        std::smatch match;
        if (std::regex_match(lineHead, match, keyValueRegex)) {
            std::string key = match[1];
            size_t valueStartPos = match.position(2);
            std::string valueSection = line.substr(valueStartPos);
            std::istringstream valuesStream(valueSection);
            std::string value;
            std::vector<std::string> values;
            while (valuesStream >> value) {
                values.push_back(value);
            }
            configMap[key] = values;
        }
    }
    infile.close();
    return configMap;
}

Json::Value readJsonConfig(const std::filesystem::path& configFile) {
    if (!std::filesystem::exists(configFile)) {
        throw std::runtime_error("Config file does not exist: " + configFile.string());
    }
    std::ifstream file(configFile);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open config file: " + configFile.string());
    }
    Json::Value root;
    file >> root;
    return root;
}


void coloradar::RadarConfig::initAntennaParams(const std::filesystem::path& antennaCfgFile) {
    auto configMap = readConfig(antennaCfgFile);
    txDistance.clear();
    txElevation.clear();
    rxDistance.clear();
    rxElevation.clear();

    auto it = configMap.find("num_rx");
    if (it != configMap.end()) {
        numRxAntennas = std::stoi(it->second[0]);
    } else {
        throw std::runtime_error("Missing num_rx in antenna config.");
    }
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
    auto configMap = readConfig(couplingCfgFile);
    couplingCalibMatrix.clear();

    auto it = configMap.find("num_doppler_bins");
    if (it != configMap.end()) {
        numDopplerBins = std::stoi(it->second[0]);
    } else {
        throw std::runtime_error("Missing num_doppler_bins in coupling config.");
    }
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
    const Json::Value& configMap = readJsonConfig(phaseFrequencyCfgFile);
    frequencyCalibMatrix.clear();
    phaseCalibMatrix.clear();
    if (!configMap.isMember("antennaCalib")) {
        throw std::runtime_error("Missing antennaCalib in phase frequency config.");
    }
    const Json::Value& config = configMap["antennaCalib"];

    if (config.isMember("frequencyCalibrationMatrix")) {
        const Json::Value& frequencyMatrix = config["frequencyCalibrationMatrix"];
        for (const auto& value : frequencyMatrix) {
            frequencyCalibMatrix.push_back(value.asDouble());
        }
    } else {
        throw std::runtime_error("Missing frequencyCalibrationMatrix in phase frequency config.");
    }
    if (config.isMember("phaseCalibrationMatrix")) {
        const Json::Value& phaseMatrix = config["phaseCalibrationMatrix"];
        if (phaseMatrix.size() % 2 != 0) {
            throw std::runtime_error("Invalid phaseCalibrationMatrix: Expecting pairs of real and imaginary values.");
        }

        for (Json::ArrayIndex i = 0; i < phaseMatrix.size(); i += 2) {
            double real = phaseMatrix[i].asDouble();
            double imag = phaseMatrix[i + 1].asDouble();
            phaseCalibMatrix.push_back(std::complex<double>(real, imag));
        }
    } else {
        throw std::runtime_error("Missing phaseCalibrationMatrix in phase frequency config.");
    }
}

void coloradar::RadarConfig::initInternalParams() {
    std::cout << "start RadarConfig::initInternalParams" << std::endl;
    numVirtualElements = numTxAntennas * numRxAntennas;
    std::cout << "numVirtualElements " << numVirtualElements << std::endl;
    if (virtualArrayMap != nullptr) {
        std::cout << "virtualArrayMap != nullptrs" << std::endl;
        delete[] virtualArrayMap;
        virtualArrayMap = nullptr;
    }
    if (rangeWindowFunc != nullptr) {
        std::cout << "rangeWindowFunc != nullptrs" << std::endl;
        delete[] rangeWindowFunc;
        rangeWindowFunc = nullptr;
    }
    if (dopplerWindowFunc != nullptr) {
        std::cout << "dopplerWindowFunc != nullptrs" << std::endl;
        delete[] dopplerWindowFunc;
        dopplerWindowFunc = nullptr;
    }
    for (size_t i = 0; i < txDistance.size(); ++i) {
        std::cout << "txDistance[i] " << txDistance[i] << std::endl;
    }
    std::cout << "cleared arrays in RadarConfig::initInternalParams" << std::endl;
    virtualArrayMap = new int[numVirtualElements * 4];
    int idx = 0;
    for (int txIdx = 0; txIdx < numTxAntennas; ++txIdx) {
        std::cout << "txIdx " << txIdx << std::endl;
        for (int rxIdx = 0; rxIdx < numRxAntennas; ++rxIdx) {
            std::cout << "txDistance[txIdx] " << txDistance[txIdx] << std::endl;
            std::cout << "rxDistance[rxIdx] " << rxDistance[rxIdx] << std::endl;
            std::cout << "txElevation[txIdx] " << txElevation[txIdx] << std::endl;
            std::cout << "rxElevation[rxIdx] " << rxElevation[rxIdx] << std::endl;
            int virtualAzIdx = txDistance[txIdx] + rxDistance[rxIdx];
            int virtualElIdx = txElevation[txIdx] + rxElevation[rxIdx];
            virtualArrayMap[idx++] = virtualAzIdx;
            virtualArrayMap[idx++] = virtualElIdx;
            virtualArrayMap[idx++] = rxIdx;
            virtualArrayMap[idx++] = txIdx;
        }
    }
    std::cout << "init virtualArrayMap in RadarConfig::initInternalParams" << std::endl;
    rangeWindowFunc = new double[numRangeBins];
    for (int i = 0; i < numRangeBins; ++i) {
        rangeWindowFunc[i] = 0.42 - 0.5 * cos(2 * M_PI * i / (numRangeBins - 1)) + 0.08 * cos(4 * M_PI * i / (numRangeBins - 1));
    }
    dopplerWindowFunc = new double[numDopplerBins];
    for (int i = 0; i < numDopplerBins; ++i) {
        dopplerWindowFunc[i] = 0.42 - 0.5 * cos(2 * M_PI * i / (numDopplerBins - 1)) + 0.08 * cos(4 * M_PI * i / (numDopplerBins - 1));
    }
}

coloradar::RadarConfig::~RadarConfig() {
    if (virtualArrayMap != nullptr) {
        delete[] virtualArrayMap;
        virtualArrayMap = nullptr;
    }
    if (rangeWindowFunc != nullptr) {
        delete[] rangeWindowFunc;
        rangeWindowFunc = nullptr;
    }
    if (dopplerWindowFunc != nullptr) {
        delete[] dopplerWindowFunc;
        dopplerWindowFunc = nullptr;
    }
}


coloradar::SingleChipConfig::SingleChipConfig(const std::filesystem::path& calibDir) {
    coloradar::internal::checkPathExists(calibDir);
    init(calibDir);
}

void coloradar::SingleChipConfig::init(const std::filesystem::path& calibDir) {
    std::cout << "start SingleChipConfig::init" << std::endl;
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
    initInternalParams();
    std::cout << "finish SingleChipConfig::init" << std::endl;
}

coloradar::CascadeConfig::CascadeConfig(const std::filesystem::path& calibDir) {
    coloradar::internal::checkPathExists(calibDir);
    init(calibDir);
}

void coloradar::CascadeConfig::init(const std::filesystem::path& calibDir) {
    std::cout << "start CascadeConfig::init" << std::endl;
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
    initInternalParams();
    std::cout << "finish CascadeConfig::init" << std::endl;
}
