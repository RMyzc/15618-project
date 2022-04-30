#include "influence.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#include <string.h>

using namespace std;

void singleNodeBFS(Graph *g, int v_id, bool visited[], int nVertices) {
    Vertex *v = g->vertices[v_id];
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

int monteCarloSimulation(Graph *g, vector<int> vertices, int numIterations) {
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
        result += visitedCount;
    }
    
    return result / numIterations;
}


int monteCarloSimulationPt(Graph *g, int* vertices, int verticesSize, int numIterations) {
    long result = 0;
    int nVertices = int(g->vertices.size());
    
    for (int round = 0; round < numIterations; round++) {
        bool visited[nVertices];
        for (int i = 0; i < nVertices; i++) visited[i] = false;

        for (int v_index = 0; v_index < verticesSize; v_index++) {
            // Search reachable neighbors
            singleNodeBFS(g, vertices[v_index], visited, nVertices);
        }

        int visitedCount = 0;
        for (int v_index = 0; v_index < nVertices; v_index++) {
            if (visited[v_index]) {
                visitedCount++;
            }
        }
        result += visitedCount;
    }
    
    return result / numIterations;
}

void compute(char *inputFilename, int nSeeds, int nMonteCarloSimulations, double prob, bool greedy, int heuristicMode) {
    /* initialize random seed: */
    srand(time(NULL));
    
    // Read input file to get the number of vertices, number of edges and all the information of the graph
    int nVertices = 0, nEdges = 0;
    Graph *g = new Graph(prob);
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
        int mode = heuristicMode;
        if (mode == BASIC) {
            vector<pair<int, int> > rank;
            for (int i = 0; i < nVertices; i++) {
                pair<int, int> p = make_pair(i, int(g->vertices[i]->neighbors.size()));
                rank.push_back(p);
            }
            sort(rank.begin(), rank.end(), cmp);
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

/* Parallel Version With Lock*/
void singleNodeBFSParallelWithLock(Graph *g, int v_id, bool visited[], omp_lock_t *locks, int nVertices) {
    Vertex *v = g->vertices[v_id];
    // Create a queue for BFS
    queue<int> q;
    
    omp_set_lock(&locks[v->id]);
    if (visited[v->id]) {
        omp_unset_lock(&locks[v->id]);
        return;
    }
    
    visited[v->id] = true;
    omp_unset_lock(&locks[v->id]);

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

                omp_set_lock(&locks[visit_id]);
                if (visited[visit_id]) {
                    omp_unset_lock(&locks[visit_id]);
                    continue;
                }
                visited[visit_id] = true;
                omp_unset_lock(&locks[visit_id]);

                q.push(visit_id);
            }
            q.pop();
        }
    }
}

/* Parallel Version */
void singleNodeBFSParallel(Graph *g, int v_id, bool visited[], int nVertices) {
    Vertex *v = g->vertices[v_id];
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


int monteCarloSimulationParallelWithLock(Graph *g, vector<int> vertices, int numIterations, int nThreads) {
    long result = 0;
    int nVertices = int(g->vertices.size());
    
    for (int round = 0; round < numIterations; round++) {
        bool visited[nVertices];
        // locks for each vertex
        omp_lock_t *locks = (omp_lock_t *)calloc(nVertices, sizeof(omp_lock_t));
        
        for (int i = 0; i < nVertices; i++) {
            visited[i] = false;
            omp_init_lock(&locks[i]);
        }

        #pragma omp parallel for
        for (int v_index = 0; v_index < int(vertices.size()); v_index++) {
            // Search reachable neighbors
            singleNodeBFSParallelWithLock(g, vertices[v_index], visited, locks, nVertices);
        }

        int visitedCount = 0;
        for (int v_index = 0; v_index < nVertices; v_index++) {
            if (visited[v_index]) {
                visitedCount++;
            }
        }
        result += visitedCount;
    }
    
    return result / numIterations;
}

int monteCarloSimulationParallel(Graph *g, vector<int> vertices, int numIterations, int nThreads) {
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
        result += visitedCount;
    }
    
    return result / numIterations;
}

void computeParallel(char *inputFilename, int nSeeds, int nMonteCarloSimulations, 
                double prob, bool greedy, int nThreads, int heuristicMode, int withLock) {
    /* Set the number of threads for the parallel region */
    omp_set_num_threads(nThreads);
    
    /* initialize random seed: */
    srand(time(NULL));
    
    // Read input file to get the number of vertices, number of edges and all the information of the graph
    int nVertices = 0, nEdges = 0;
    Graph *g = new Graph(prob);
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

        // To reduce the malloc memory of allPermutation
        int assign[GREEDY_DIVIDE];
        memset(&assign, 0, sizeof(int) * GREEDY_DIVIDE);
        int i = 0;
        for (; i < numPermutations; i++) {
            assign[i % GREEDY_DIVIDE]++;
        }
        i--;
        int lineSize = assign[i % GREEDY_DIVIDE];
        
        // int allPermutations[numPermutations][nSeeds];
        int** allPermutations = new int*[lineSize];
        for (int pt = 0; pt < lineSize; pt++) {
            allPermutations[pt] = new int[nSeeds];
        }
        // memset(&allPermutations, 0, sizeof(int) * numPermutations * nSeeds);
        
        for (int round = 0; round < GREEDY_DIVIDE; round++) {
            int permutationsCnt = 0;
            int curSize = 0;

            do {
                if (curSize == assign[round]) break;
                int cnt = 0;
                for (int i = 0; i < nVertices; ++i) {
                    if (vec[i]) {
                        allPermutations[permutationsCnt][cnt++] = i;
                    }
                }
                permutationsCnt++;
                curSize++;
            } while (prev_permutation(vec.begin(), vec.end()));

            int i, nThreads;
            #pragma omp parallel default(shared) private(i, nThreads) 
            {
                i = omp_get_thread_num();
                nThreads = omp_get_num_threads();
                for (i = 0; i < curSize; i += nThreads) {
                    // vector<int> perm(allPermutations[i], allPermutations[i] + nSeeds);
                    // int cur = monteCarloSimulation(g, perm, nMonteCarloSimulations);
                    
                    int cur = monteCarloSimulationPt(g, allPermutations[i], nSeeds, nMonteCarloSimulations);
                    #pragma omp critical
                    {
                        if (cur > maxval) {
                            maxval = cur;
                            seeds.assign(allPermutations[i], allPermutations[i] + nSeeds);
                        }
                    }
                }
            }
        }
        printf("maxval = %d\n", maxval);

        for(int pt = 0; pt < lineSize; pt++) {
            delete []allPermutations[pt];
        }
        delete []allPermutations;
    } else {
        // Heuristic
        int mode = heuristicMode;
        if (mode == BASIC) {
            vector<pair<int, int> > rank;
            for (int i = 0; i < nVertices; i++) {
                pair<int, int> p = make_pair(i, int(g->vertices[i]->neighbors.size()));
                rank.push_back(p);
            }
            sort(rank.begin(), rank.end(), cmp);
            vector<int> seeds;
            for (int i = 0; i < nSeeds; i++) {
                seeds.push_back(rank[i].first);
            }
            
            int result = 0;
            if (withLock) {
                result = monteCarloSimulationParallelWithLock(g, seeds, nMonteCarloSimulations, nThreads);
            } else {
                result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            }
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
            int result = 0;
            if (withLock) {
                result = monteCarloSimulationParallelWithLock(g, seeds, nMonteCarloSimulations, nThreads);
            } else {
                result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            }
            printf("Minus-1 heuristic result = %d\n", result);
        } else if (mode == DEGREEDISCOUNT) {
            unordered_map<int, int> id2degree;
            // #pragma omp parallel for
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

            int result = 0;
            if (withLock) {
                result = monteCarloSimulationParallelWithLock(g, seeds, nMonteCarloSimulations, nThreads);
            } else {
                result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            }
            printf("Degree Discount heuristic result = %d\n", result);
        } else if (mode == DEGREEDISCOUNTPARALLEL) {
            // unordered_map<int, int> id2degree;
            unordered_map<int, pair<omp_lock_t*, int>> id2degree;
            // #pragma omp parallel for shared(id2degree)
            for (int i = 0; i < nVertices; i++) {
                omp_lock_t* lock = new omp_lock_t();
                omp_init_lock(lock);
                pair<omp_lock_t*, int> newVertice = {lock, int(g->vertices[i]->neighbors.size())};
                id2degree[g->vertices[i]->id] = newVertice;
                // id2degree[g->vertices[i]->id] = int(g->vertices[i]->neighbors.size());
            }

            // this map is sorted by descending value of degree
            // map<int, unordered_set<int>, greater<int> > degree2ids;
            map<int, pair<omp_lock_t*, unordered_set<int>>, greater<int> > degree2ids;
            for (int i = 0; i <= g->vertices.size(); i++) {
                unordered_set<int> uset;

                omp_lock_t* lock = new omp_lock_t();
                omp_init_lock(lock);
                pair<omp_lock_t*, unordered_set<int>> newVertice = {lock, uset};

                degree2ids[i] = newVertice; 
            }

            for (auto it = id2degree.begin(); it != id2degree.end(); it++) {
                degree2ids[it->second.second].second.insert(it->first);
            }

            unordered_set<int> seedsSet;  
            int totalCount = 0;          
            for (int i = 0; i < nSeeds; i += nThreads) {
                int count = 0;
                vector<int> tempSeeds;
                // get # nThreads and use them to update neighbors' degree in parallel
                for (auto it1 = degree2ids.begin(); it1 != degree2ids.end(); it1++) {
                    unordered_set<int> uset = degree2ids[it1->first].second;
                    vector<int> innerSeeds;
                    for (auto it2 = uset.begin(); it2 != uset.end(); it2++) {
                        int id = *it2;
                        seedsSet.insert(id);
                        tempSeeds.push_back(id);
                        innerSeeds.push_back(id);
                        id2degree.erase(id);
                        count++;
                        totalCount++;
                        if (count == nThreads || totalCount == nSeeds) break;
                    }
                    for (int ii = 0; ii < innerSeeds.size(); ii++) {
                        degree2ids[it1->first].second.erase(innerSeeds[ii]);
                    }
                    if (count == nThreads || totalCount == nSeeds) break;
                }
                
                #pragma omp parallel for
                for (int currSeed = 0; currSeed < tempSeeds.size(); currSeed++) {
                    Vertex* v = g->vertices[currSeed];
                    for (int j = 0; j < int(v->neighbors.size()); j++) {
                        int neighborId = v->neighbors[j];
                        if (id2degree.find(neighborId) == id2degree.end()) continue;

                        int originalDegree = int(g->vertices[neighborId]->neighbors.size());

                        int neighborSeedsCnt = 0;
                        for (int k = 0; k < int(g->vertices[neighborId]->neighbors.size()); k++) {
                            if (seedsSet.find(g->vertices[neighborId]->neighbors[k]) != seedsSet.end()) {
                                neighborSeedsCnt++;
                            }
                        }

                        // ddv = dv - [2tv + (dv - tv)tvp]
                        int updateTo = originalDegree - (2 * neighborSeedsCnt + 
                            (originalDegree - neighborSeedsCnt) * neighborSeedsCnt * g->prob);
                        if (updateTo < 0) {
                            updateTo = 0;
                        }

                        // update id2degree
                        int liveDegree;
                        omp_set_lock(id2degree[neighborId].first);
                        liveDegree = id2degree[neighborId].second;
                        id2degree[neighborId].second = updateTo;
                        omp_unset_lock(id2degree[neighborId].first);

                        // update degree2ids
                        // Get original degree
                        omp_set_lock(degree2ids[liveDegree].first);
                        degree2ids[liveDegree].second.erase(neighborId);
                        omp_unset_lock(degree2ids[liveDegree].first);
                        // Update new degree
                        omp_set_lock(degree2ids[updateTo].first);
                        degree2ids[updateTo].second.insert(neighborId);
                        omp_unset_lock(degree2ids[updateTo].first);
                    } 
                }
            }

            vector<int> seeds;
            seeds.insert(seeds.end(), seedsSet.begin(), seedsSet.end());

            int result = 0;
            if (withLock) {
                result = monteCarloSimulationParallelWithLock(g, seeds, nMonteCarloSimulations, nThreads);
            } else {
                result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            }
            printf("Parallel Degree Discount Heuristic result = %d\n", result);
        }
    }
}