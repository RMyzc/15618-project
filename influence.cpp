#include "influence.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <chrono>
#include <time.h>
#include <math.h>
#include <limits.h>

using namespace std;

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

void singleNodeBFS(graph_t *g, int v_id, bool visited[], int nVertices) {
    vertex_t *v = g->vertices[v_id];
    if (visited[v->id]) {
        return;
    }

    // Create a queue for BFS
    queue<int> q;

    visited[v->id] = true;
    q.push(v->id);
 
    while(!q.empty())
    {
        int size = q.size();
 
        for (int i = 0; i < size; i++) {
            int cur = q.front();
            for (int j = 0; j < int(g->vertices[cur]->neighbors.size()); j++) {
                bool isVisit = ((float)rand() / (float) RAND_MAX) <= g->prob;
                if (!isVisit) {
                    continue;
                }

                int visit_id = g->vertices[cur]->neighbors[j];
                if (visited[visit_id]) {
                    continue;
                }
                q.push(visit_id);
                visited[visit_id] = true;
            }
            q.pop();
        }
    }
    
}

long monteCarloSimulation(graph_t *g, vector<int> vertices, int numIterations) {
    long result = 0;
    int nVertices = int(g->vertices.size());
    
    for (int round = 0; round < numIterations; round++) {
        bool visited[nVertices];
        for (int i = 0; i < nVertices; i++) visited[i] = false;

        for (int v_index = 0; v_index < int(vertices.size()); v_index++) {
            // Search reachable neighbors
            singleNodeBFS(g, vertices[v_index], visited, nVertices);
        }

        int visitedCount = 0;
        for (int v_index = 0; v_index < nVertices; v_index++) {
            if (visited[v_index]) {
                visitedCount++;
            }
        }
        // printf("round %d, visitedCount %d\n", round, visitedCount);
        result += visitedCount;
    }
    
    return result / numIterations;
}

void compute(char *inputFilename, int nSeeds, int nMonteCarloSimulations, double prob, bool greedy) {
    /* initialize random seed: */
    srand(time(NULL));
    
    // Read input file to get the number of vertices, number of edges and all the information of the graph
    int nVertices = 0, nEdges = 0;
    graph_t *g = new graph_t(prob);
    readInput(inputFilename, &nVertices, &nEdges, &g);

    if (greedy) {
        int maxval = INT_MIN;
        vector<int> seeds, selectedVertices;
        vector<bool> vec(nVertices, false);
        fill(vec.begin(), vec.begin() + nSeeds, true);

        do {
            selectedVertices.clear();
            for (int i = 0; i < nVertices; ++i) {
                if (vec[i]) {
                    selectedVertices.push_back(i);
                }
            }
            int cur = monteCarloSimulation(g, selectedVertices, nMonteCarloSimulations);
            if (cur > maxval) {
                maxval = cur;
                seeds.assign(selectedVertices.begin(), selectedVertices.end());
            }
        } while (prev_permutation(vec.begin(), vec.end()));
        printf("maxval = %d\n", maxval);
    } else {
        // Heuristic
    }

}