#include "coloradar_tools.h"

#include <unordered_map>
#include <tuple>
#include <functional>


POINT_CLOUD_REGISTER_POINT_STRUCT (coloradar::RadarPoint,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (float, doppler, doppler))


template <>
struct std::hash<std::tuple<float, float, float>> {
    std::size_t operator()(const std::tuple<float, float, float>& key) const {
        auto roundToPrecision = [](float value, float precision) {
            return std::round(value / precision) * precision;
        };
        std::size_t h1 = std::hash<float>{}(roundToPrecision(std::get<0>(key), 0.01f));
        std::size_t h2 = std::hash<float>{}(roundToPrecision(std::get<1>(key), 0.01f));
        std::size_t h3 = std::hash<float>{}(roundToPrecision(std::get<2>(key), 0.01f));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};
inline std::tuple<float, float, float> makeKey(float x, float y, float z) {
    return std::make_tuple(x, y, z);
}

pcl::PointCloud<coloradar::RadarPoint> coloradar::heatmapToPointcloud(const std::vector<float>& heatmap, coloradar::RadarConfig* config, const float& intensityThresholdPercent) {
    if (intensityThresholdPercent < 0 or intensityThresholdPercent >= 100)
        throw std::runtime_error("Invalid intensityThresholdPercent: expected value in [0; 100), got " + std::to_string(intensityThresholdPercent));
    float maxIntensity = 0;
    pcl::PointCloud<coloradar::RadarPoint> cloud;
    std::unordered_map<std::tuple<float, float, float>, RadarPoint, std::hash<std::tuple<float, float, float>>> pointMap;

    for (int azIdx = 0; azIdx < config->numAzimuthBeams; azIdx++) {
        for (int rangeIdx = 10; rangeIdx < config->numPosRangeBins; rangeIdx++) {
            float maxElIntensity = 0.0f;
            float maxElDoppler = 0.0f;
            int maxElBin = 0;

            for (int elIdx = 0; elIdx < config->numElevationBeams; elIdx++) {
                int angleIdx = azIdx + config->numAzimuthBeams * elIdx;
                int outIdx = 2 * (rangeIdx + config->numPosRangeBins * angleIdx);
                if (heatmap[outIdx] > maxElIntensity) {
                    maxElIntensity = heatmap[outIdx];
                    maxElDoppler = heatmap[outIdx + 1];
                    maxElBin = elIdx;
                }
            }
            if (maxElIntensity > maxIntensity)
                maxIntensity = maxElIntensity;
            double range = rangeIdx * config->rangeBinWidth;
            Eigen::Vector3f location = coloradar::internal::sphericalToCartesian(config->azimuthAngles[azIdx], config->elevationAngles[maxElBin], range);
            RadarPoint point;
            point.x = location.x();
            point.y = location.y();
            point.z = location.z();
            point.intensity = maxElIntensity;
            point.doppler = maxElDoppler;
            auto key = makeKey(point.x, point.y, point.z);
            auto it = pointMap.find(key);
            if (it != pointMap.end()) {
                if (it->second.intensity < point.intensity) {
                    it->second = point;
                }
            } else {
                pointMap[key] = point;
            }
        }
    }
    float intensityThreshold = maxIntensity * intensityThresholdPercent / 100;
    for (const auto& kv : pointMap) {
        if (kv.second.intensity >= intensityThreshold)
            cloud.push_back(kv.second);
    }
    return cloud;
}
