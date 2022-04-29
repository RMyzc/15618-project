#include <vector>
#include <iostream>
#include <fstream>

#include "vertex.h"
#include "graph.h"
#include "util.h"

using namespace std;

const int GREEDY_DIVIDE = 64;

enum heuristicMode {
    BASIC,
    MINUSONE,
    DEGREEDISCOUNT,
};

void compute(char *inputFilename, int nSeeds, int nMonteCarloSimulations, double prob, bool greedy);
void computeParallel(char *inputFilename, int nSeeds, int nMonteCarloSimulations, double prob, bool greedy, int nThreads);