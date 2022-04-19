#include "influence.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#include <string.h>

using namespace std;

static bool cmp(const pair<int, int>& p1, const pair<int, int>& p2) {
    return p1.second > p2.second;
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

void singleNodeBFS(graph_t *g, int v_id, bool visited[], int nVertices) {
    vertex_t *v = g->vertices[v_id];
    if (visited[v->id]) {
        return;
    }

    // Create a queue for BFS
    queue<int> q;

    visited[v->id] = true;
    q.push(v->id);

    unsigned int seed = (unsigned int) time(NULL);
 
    while(!q.empty())
    {
        int size = q.size();
 
        for (int i = 0; i < size; i++) {
            int cur = q.front();
            for (int j = 0; j < int(g->vertices[cur]->neighbors.size()); j++) {
                bool isVisit = (rand_r(&seed) / (float) RAND_MAX) <= g->prob;
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

int monteCarloSimulation(graph_t *g, vector<int> vertices, int numIterations) {
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
        int mode = DEGREEDISCOUNT;
        if (mode == BASIC) {
            vector<pair<int, int> > rank;
            for (int i = 0; i < nVertices; i++) {
                pair<int, int> p = make_pair(i, int(g->vertices[i]->neighbors.size()));
                rank.push_back(p);
            }
            sort(rank.begin(), rank.end(), cmp);
            // for (int i = 0; i < nVertices; i++) {
            //     printf("%d %d\n", rank[i].first, rank[i].second);
            // }
            vector<int> seeds;
            for (int i = 0; i < nSeeds; i++) {
                seeds.push_back(rank[i].first);
            }
            int result = monteCarloSimulation(g, seeds, nMonteCarloSimulations);
            printf("Basic heuristic result = %d\n", result);
        } else if (mode == MINUSONE) {
            unordered_map<int, int> id2degree;
            for (int i = 0; i < nVertices; i++) {
                id2degree[g->vertices[i]->id] = int(g->vertices[i]->neighbors.size());
            }

            vector<int> seeds;
            for (int i = 0; i < nSeeds; i++) {
                int maxId = -1;
                int maxDegree = INT_MIN;
                for (auto it = id2degree.begin(); it != id2degree.end(); it++) {
                    if (it->second > maxDegree) {
                        maxId = it->first;
                        maxDegree = it->second;
                    }
                }
                seeds.push_back(maxId);
                id2degree.erase(maxId);
                for (int j = 0; j < int(g->vertices[maxId]->neighbors.size()); j++) {
                    id2degree[g->vertices[maxId]->neighbors[j]] -= 1;
                }
            }
            int result = monteCarloSimulation(g, seeds, nMonteCarloSimulations);
            printf("Minus-1 heuristic result = %d\n", result);
        } else if (mode == DEGREEDISCOUNT) {
            unordered_map<int, int> id2degree;
            for (int i = 0; i < nVertices; i++) {
                id2degree[g->vertices[i]->id] = int(g->vertices[i]->neighbors.size());
            }

            unordered_set<int> seedsSet;            
            for (int i = 0; i < nSeeds; i++) {
                int maxId = -1;
                int maxDegree = INT_MIN;
                for (auto it = id2degree.begin(); it != id2degree.end(); it++) {
                    if (it->second > maxDegree) {
                        maxId = it->first;
                        maxDegree = it->second;
                    }
                }
                seedsSet.insert(maxId);
                id2degree.erase(maxId);
                
                // ddv = dv - [2tv + (dv - tv)tvp]
                for (int j = 0; j < int(g->vertices[maxId]->neighbors.size()); j++) {
                    if (id2degree.find(g->vertices[maxId]->neighbors[j]) == id2degree.end()) continue;
                    
                    int neighborId = g->vertices[maxId]->neighbors[j];
                    int originalDegree =  int(g->vertices[neighborId]->neighbors.size());
                    
                    int neighborSeedsCnt = 0;
                    for (int k = 0; k < int(g->vertices[neighborId]->neighbors.size()); k++) {
                        if (seedsSet.find(g->vertices[neighborId]->neighbors[k]) != seedsSet.end()) {
                            neighborSeedsCnt++;
                        }
                    }
                    id2degree[g->vertices[maxId]->neighbors[j]] = 
                        originalDegree - (2 * neighborSeedsCnt + 
                        (originalDegree - neighborSeedsCnt) * neighborSeedsCnt * g->prob);
                }
            }

            vector<int> seeds;
            seeds.insert(seeds.end(), seedsSet.begin(), seedsSet.end());
            int result = monteCarloSimulation(g, seeds, nMonteCarloSimulations);
            printf("Degree Discount heuristic result = %d\n", result);

        }

    }

}

/* Parallel Version */
void singleNodeBFSParallel(graph_t *g, int v_id, bool visited[], int nVertices) {
    vertex_t *v = g->vertices[v_id];
    if (visited[v->id]) {
        return;
    }

    // Create a queue for BFS
    queue<int> q;

    visited[v->id] = true;
    q.push(v->id);

    unsigned int seed = (unsigned int) time(NULL);
 
    while(!q.empty())
    {
        int size = q.size();
 
        for (int i = 0; i < size; i++) {
            int cur = q.front();
            for (int j = 0; j < int(g->vertices[cur]->neighbors.size()); j++) {
                bool isVisit = (rand_r(&seed) / (float) RAND_MAX) <= g->prob;
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

int monteCarloSimulationParallel(graph_t *g, vector<int> vertices, int numIterations, int nThreads) {
    long result = 0;
    int nVertices = int(g->vertices.size());
    
    #pragma omp parallel for reduction(+ : result)
    for (int round = 0; round < numIterations; round++) {
        bool visited[nVertices];
        for (int i = 0; i < nVertices; i++) visited[i] = false;

        for (int v_index = 0; v_index < int(vertices.size()); v_index++) {
            // Search reachable neighbors
            singleNodeBFSParallel(g, vertices[v_index], visited, nVertices);
        }

        int visitedCount = 0;
        for (int v_index = 0; v_index < nVertices; v_index++) {
            if (visited[v_index]) {
                visitedCount++;
            }
        }
        // printf("thread %d, round %d, visitedCount %d\n", omp_get_thread_num(), round, visitedCount);
        result += visitedCount;
    }
    
    return result / numIterations;
}

void computeParallel(char *inputFilename, int nSeeds, int nMonteCarloSimulations, 
                double prob, bool greedy, int nThreads) {
    /* Set the number of threads for the parallel region */
    omp_set_num_threads(nThreads);
    
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

        int numPermutations = 0;
        do {
            numPermutations++;
        } while (prev_permutation(vec.begin(), vec.end()));

        // int allPermutations[numPermutations][nSeeds];
        int** allPermutations = new int*[numPermutations];
        for (int pt = 0; pt < numPermutations; pt++) {
            allPermutations[pt] = new int[nSeeds];
        }
        // memset(&allPermutations, 0, sizeof(int) * numPermutations * nSeeds);
        
        int permutationsCnt = 0;

        do {
            // vector<int> cur(nSeeds, 0);
            // allPermutations[permutationsCnt++] = cur;
            int cnt = 0;
            for (int i = 0; i < nVertices; ++i) {
                if (vec[i]) {
                    allPermutations[permutationsCnt][cnt++] = i;
                }
            }
            permutationsCnt++;
        } while (prev_permutation(vec.begin(), vec.end()));

        #pragma omp parallel for
        for (int i = 0; i < numPermutations; i++) {
            vector<int> perm(allPermutations[i], allPermutations[i] + nSeeds);
            
            int cur = monteCarloSimulation(g, perm, nMonteCarloSimulations);
            #pragma omp critical
            {
                if (cur > maxval) {
                    maxval = cur;
                    seeds.assign(perm.begin(), perm.end());
                }
            }
        }
        printf("maxval = %d\n", maxval);

        for(int pt = 0; pt < numPermutations; pt++) {
            delete []allPermutations[pt];
        }
        delete []allPermutations;
    } else {
        // Heuristic
        int mode = DEGREEDISCOUNT;
        if (mode == BASIC) {
            vector<pair<int, int> > rank;
            for (int i = 0; i < nVertices; i++) {
                pair<int, int> p = make_pair(i, int(g->vertices[i]->neighbors.size()));
                rank.push_back(p);
            }
            sort(rank.begin(), rank.end(), cmp);
            // for (int i = 0; i < nVertices; i++) {
            //     printf("%d %d\n", rank[i].first, rank[i].second);
            // }
            vector<int> seeds;
            for (int i = 0; i < nSeeds; i++) {
                seeds.push_back(rank[i].first);
            }
            int result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            printf("Basic heuristic result = %d\n", result);
        } else if (mode == MINUSONE) {
            unordered_map<int, int> id2degree;
            for (int i = 0; i < nVertices; i++) {
                id2degree[g->vertices[i]->id] = int(g->vertices[i]->neighbors.size());
            }

            vector<int> seeds;
            for (int i = 0; i < nSeeds; i++) {
                int maxId = -1;
                int maxDegree = INT_MIN;
                for (auto it = id2degree.begin(); it != id2degree.end(); it++) {
                    if (it->second > maxDegree) {
                        maxId = it->first;
                        maxDegree = it->second;
                    }
                }
                seeds.push_back(maxId);
                id2degree.erase(maxId);
                for (int j = 0; j < int(g->vertices[maxId]->neighbors.size()); j++) {
                    id2degree[g->vertices[maxId]->neighbors[j]] -= 1;
                }
            }
            int result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            printf("Minus-1 heuristic result = %d\n", result);
        } else if (mode == DEGREEDISCOUNT) {
            unordered_map<int, int> id2degree;
            for (int i = 0; i < nVertices; i++) {
                id2degree[g->vertices[i]->id] = int(g->vertices[i]->neighbors.size());
            }

            unordered_set<int> seedsSet;            
            for (int i = 0; i < nSeeds; i++) {
                int maxId = -1;
                int maxDegree = INT_MIN;
                for (auto it = id2degree.begin(); it != id2degree.end(); it++) {
                    if (it->second > maxDegree) {
                        maxId = it->first;
                        maxDegree = it->second;
                    }
                }
                seedsSet.insert(maxId);
                id2degree.erase(maxId);
                
                // ddv = dv - [2tv + (dv - tv)tvp]
                for (int j = 0; j < int(g->vertices[maxId]->neighbors.size()); j++) {
                    if (id2degree.find(g->vertices[maxId]->neighbors[j]) == id2degree.end()) continue;
                    
                    int neighborId = g->vertices[maxId]->neighbors[j];
                    int originalDegree =  int(g->vertices[neighborId]->neighbors.size());
                    
                    int neighborSeedsCnt = 0;
                    for (int k = 0; k < int(g->vertices[neighborId]->neighbors.size()); k++) {
                        if (seedsSet.find(g->vertices[neighborId]->neighbors[k]) != seedsSet.end()) {
                            neighborSeedsCnt++;
                        }
                    }
                    id2degree[g->vertices[maxId]->neighbors[j]] = 
                        originalDegree - (2 * neighborSeedsCnt + 
                        (originalDegree - neighborSeedsCnt) * neighborSeedsCnt * g->prob);
                }
            }

            vector<int> seeds;
            seeds.insert(seeds.end(), seedsSet.begin(), seedsSet.end());
            int result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            printf("Degree Discount heuristic result = %d\n", result);
        }
    }
}