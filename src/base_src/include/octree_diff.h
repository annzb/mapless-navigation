#include <octomap/octomap.h>

#ifndef OCTREE_DIFF_H
#define OCTREE_DIFF_H

struct DiffNode {
    octomap::point3d coords;
    float logOdds;
    float logOddsDiff;

    DiffNode(const octomap::point3d& coords, float logOdds, float logOddsDiff)
        : coords(coords), logOdds(logOdds), logOddsDiff(logOddsDiff) {}
};

std::vector<DiffNode> calcOctreeDiff(const octomap::OcTree& tree1, const octomap::OcTree& tree2, const double eps = 0.001);

#endif
