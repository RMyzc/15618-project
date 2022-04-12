#include "influence.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

void compute(char *inputFilename, int nSeeds, int nMonteCarloSimulations, double prob, double *startTime, double *endTime) {
    // Read input file to get the number of vertices, number of edges and all the information of the graph
    int nVertices = 0, nEdges = 0;
    graph_t *g = new graph_t(prob);
    readInput(inputFilename, &nVertices, &nEdges, &g);
}

void readInput(char *inputFilename, int *nVertices, int *nEdges, graph_t **g) {
    ifstream infile(inputFilename);
    // Read the first line of the input data and put them into the graph
    infile >> *nVertices >> *nEdges;
    for (int i = 0; i < *nVertices; i++) {
        vertex_t *v = new vertex(i);
        (*g)->vertices.push_back(v);
    }
    // Read the neighbors from the input data
    int from, to;
    while (infile >> from >> to) {
        (*g)->vertices[from]->neighbors.push_back(to);
        (*g)->vertices[to]->neighbors.push_back(from);
    }
}