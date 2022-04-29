#include <vector>
#include <iostream>
#include <fstream>

#include "vertex.h"
#include "graph.h"
#include "util.h"

using namespace std;

const int GREEDY_DIVIDE = 64;

enum heuristicModeEnum {
    BASIC,
    MINUSONE,
    DEGREEDISCOUNT,
};

void compute(char *inputFilename, int nSeeds, int nMonteCarloSimulations, 
            double prob, bool greedy, int heuristicMode);
void computeParallel(char *inputFilename, int nSeeds, int nMonteCarloSimulations, 
            double prob, bool greedy, int nThreads, int heuristicMode, int withLock);