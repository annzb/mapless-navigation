#include <octomap/octomap.h>
#include <unordered_set>
#include <cmath>
#include <vector>

#include "octree_diff.h"


struct Point3DHash {
    std::size_t operator()(const octomap::point3d& point) const {
        std::size_t seed = 0;
        std::hash<float> hasher;
        seed ^= hasher(point.x()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(point.y()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(point.z()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};


std::vector<DiffNode> calcOctreeDiff(
    const octomap::OcTree& tree1,
    const octomap::OcTree& tree2,
    double nodeDiffEps
) {
    std::vector<DiffNode> diffNodes;
    std::unordered_set<octomap::point3d, Point3DHash> checkedNodes;

    for (auto it = tree2.begin_leafs(), end = tree2.end_leafs(); it != end; ++it) {
        octomap::point3d nodeCoords = it.getCoordinate();
        float logOdds2 = it->getLogOdds();
        auto nodeInTree1 = tree1.search(nodeCoords);
        float logOdds1 = nodeInTree1 ? nodeInTree1->getLogOdds() : 0.0f;
        float occupancyUpdate = logOdds2 - logOdds1;
        if (std::fabs(occupancyUpdate) >= nodeDiffEps) {
            diffNodes.emplace_back(nodeCoords, logOdds2, occupancyUpdate);
        }
        checkedNodes.insert(nodeCoords);
    }

    for (auto it = tree1.begin_leafs(), end = tree1.end_leafs(); it != end; ++it) {
        octomap::point3d nodeCoords = it.getCoordinate();
        if (checkedNodes.find(nodeCoords) == checkedNodes.end()) {
            float occupancyUpdate = -it->getLogOdds();
            if (std::fabs(occupancyUpdate) >= nodeDiffEps) {
                diffNodes.emplace_back(nodeCoords, std::numeric_limits<float>::quiet_NaN(), occupancyUpdate);
            }
        }
    }
    return diffNodes;
}
