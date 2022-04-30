/**
 * CMU 15-618 Parallel Computer Architecture and Programming
 * Parallel Influence Maximization in Social Networks
 * 
 * Author: Zican Yang(zicany), Yuling Wu(yulingw)
 */
 
#include <stdio.h>
#include <iostream>

using namespace std;

/**
 * Sorting the out-degrees in descending order in heuristic algorithm.
 */
static bool cmp(const pair<int, int>& p1, const pair<int, int>& p2) {
    return p1.second > p2.second;
}

/**
 * Read input social network representation from a file
 */
static void readInput(char *inputFilename, int *nVertices, int *nEdges, Graph **g) {
    ifstream infile(inputFilename);

    // Read the first line of the input data and put all the vertices into the graph
    infile >> *nVertices >> *nEdges;
    for (int i = 0; i < *nVertices; i++) {
        Vertex *v = new Vertex(i);
        (*g)->vertices.push_back(v);
    }

    // Read the neighbors of vertices from the input data
    int from, to;
    while (infile >> from >> to) {
        (*g)->vertices[from]->neighbors.push_back(to);
        (*g)->vertices[to]->neighbors.push_back(from);
    }
}