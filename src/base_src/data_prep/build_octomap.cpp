#include "coloradar_tools.h"
#include <unordered_map>


namespace fs = std::filesystem;


std::unordered_map<std::string, std::string> parseArguments(int argc, char** argv) {
    std::unordered_map<std::string, std::string> arguments;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("=") != std::string::npos) {
            auto pos = arg.find("=");
            std::string key = arg.substr(0, pos);
            std::string value = arg.substr(pos + 1);
            arguments[key] = value;
        }
    }
    return arguments;
}


int main(int argc, char** argv) {
    auto args = parseArguments(argc, argv);
    std::string coloradarDir = (args.find("coloradarDir") != args.end())
                               ? args["coloradarDir"]
                               : (argc > 1 ? argv[1] : "");
    std::string runName = (args.find("runName") != args.end())
                          ? args["runName"]
                          : (argc > 2 ? argv[2] : "");
    if (coloradarDir.empty()) {
        std::cerr << "Usage: " << argv[0] << " <coloradarDir> [<runName>] [mapResolution=<meters>] [verticalFov=<degrees>] [horizontalFov=<degrees>] [range=<meters>]" << std::endl;
        return -1;
    }
    double mapResolution = args.find("mapResolution") != args.end() ? std::stod(args["mapResolution"]) : 0.1;
    double verticalFov = args.find("verticalFov") != args.end() ? std::stod(args["verticalFov"]) : 180.0;
    double horizontalFov = args.find("horizontalFov") != args.end() ? std::stod(args["horizontalFov"]) : 360.0;
    double range = args.find("range") != args.end() ? std::stod(args["range"]) : 0.0;

    coloradar::ColoradarDataset dataset(coloradarDir);
    std::vector<std::string> targetRuns;
    if (!runName.empty()) {
        targetRuns.push_back(runName);
    }
    dataset.createMaps(mapResolution, horizontalFov, verticalFov, range, targetRuns);

    return 0;
}
