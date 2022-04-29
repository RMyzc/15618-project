#include <stdio.h>
#include <iostream>

using namespace std;

static bool cmp(const pair<int, int>& p1, const pair<int, int>& p2) {
    return p1.second > p2.second;
}

static void readInput(char *inputFilename, int *nVertices, int *nEdges, Graph **g) {
    ifstream infile(inputFilename);
    // Read the first line of the input data and put them into the graph
    infile >> *nVertices >> *nEdges;
    for (int i = 0; i < *nVertices; i++) {
        Vertex *v = new Vertex(i);
        (*g)->vertices.push_back(v);
    }
    // Read the neighbors from the input data
    int from, to;
    while (infile >> from >> to) {
        (*g)->vertices[from]->neighbors.push_back(to);
        (*g)->vertices[to]->neighbors.push_back(from);
    }
}