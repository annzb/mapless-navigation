#include <octomap/octomap.h>
#include <unordered_set>
#include <cmath>
#include <functional>

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

std::pair<octomap::OcTree, octomap::OcTree> calcOctreeDiff(
    const octomap::OcTree& tree1,
    const octomap::OcTree& tree2,
    const double updateEps
) {
    double resolution = tree1.getResolution();
    octomap::OcTree diffTree(resolution);
    octomap::OcTree updateTree(resolution);

    std::unordered_set<octomap::point3d, Point3DHash> checkedNodes;

    for (auto it = tree2.begin_leafs(), end = tree2.end_leafs(); it != end; ++it) {
        double logOdds = it->getLogOdds();
        octomap::point3d nodeCoords = it.getCoordinate();
        bool nodeOccupied = it->getValue();

        auto nodeInTree1 = tree1.search(nodeCoords);
        double occupancyUpdate = logOdds - (nodeInTree1 ? nodeInTree1->getLogOdds() : 0.0);

        if (std::fabs(occupancyUpdate) >= updateEps) {
            updateTree.updateNode(nodeCoords, nodeOccupied)->setLogOdds(logOdds);
            diffTree.updateNode(nodeCoords, nodeOccupied)->setLogOdds(occupancyUpdate);
        }

        checkedNodes.insert(nodeCoords);
    }

    for (auto it = tree1.begin_leafs(), end = tree1.end_leafs(); it != end; ++it) {
        octomap::point3d nodeCoords = it.getCoordinate();
        if (checkedNodes.find(nodeCoords) == checkedNodes.end()) {
            double occupancyUpdate = -(it->getLogOdds());
            if (std::fabs(occupancyUpdate) >= updateEps) {
                diffTree.updateNode(nodeCoords, false)->setLogOdds(occupancyUpdate);
            }
        }
    }

    return std::make_pair(updateTree, diffTree);
}

