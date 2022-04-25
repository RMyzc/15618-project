#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

const int GREEDY_DIVIDE = 64;

enum heuristicMode {
    BASIC,
    MINUSONE,
    DEGREEDISCOUNT,
};

typedef struct vertex {
    int id;
    vector<int> neighbors;

    vertex(int index) {
        id = index;
        neighbors.clear();
    }
} vertex_t;

typedef struct graph {
    vector<vertex_t*> vertices;
    double prob; // propagation probability

    graph() {
        vertices.clear();
        prob = 0.1;
    }

    graph(double p) {
        vertices.clear();
        prob = p;
    }
} graph_t;

void compute(char *inputFilename, int nSeeds, int nMonteCarloSimulations, double prob, bool greedy);
void computeParallel(char *inputFilename, int nSeeds, int nMonteCarloSimulations, double prob, bool greedy, int nThreads);