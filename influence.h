/**
 * CMU 15-618 Parallel Computer Architecture and Programming
 * Parallel Influence Maximization in Social Networks
 * 
 * Author: Zican Yang(zicany), Yuling Wu(yulingw)
 */

#include <vector>
#include <iostream>
#include <fstream>

#include "vertex.h"
#include "graph.h"
#include "util.h"

using namespace std;

// Divide the total malloc memory by this number to reduce malloc overhead
const int GREEDY_DIVIDE = 64;

// Four kinds of heuristic algorithms
enum heuristicModeEnum {
    BASIC,
    MINUSONE,
    DEGREEDISCOUNT,
    DEGREEDISCOUNTPARALLEL,
};

void compute(char *inputFilename, int nSeeds, int nMonteCarloSimulations, 
            double prob, bool greedy, int heuristicMode);
void computeParallel(char *inputFilename, int nSeeds, int nMonteCarloSimulations, 
            double prob, bool greedy, int nThreads, int heuristicMode, int withLock);