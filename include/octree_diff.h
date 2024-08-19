#include <octomap/octomap.h>

#ifndef OCTREE_DIFF_H
#define OCTREE_DIFF_H

std::pair<octomap::OcTree, octomap::OcTree> calcOctreeDiff(const octomap::OcTree& tree1, const octomap::OcTree& tree2, const double eps = 0.001);

#endif
