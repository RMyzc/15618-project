/**
 * CMU 15-618 Parallel Computer Architecture and Programming
 * Parallel Influence Maximization in Social Networks
 * 
 * Author: Zican Yang(zicany), Yuling Wu(yulingw)
 */

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

/**
 * Given a node and a visited memo, do a breadth first search in the graph
 * The node will try to activate all its unvisited neighbors according to propagation probability
 * Activated nodes will also try to active its neighbors, recursively until no more can be activated
 */
void singleNodeBFS(Graph *g, int v_id, bool visited[], int nVertices) {
    Vertex *v = g->vertices[v_id];
    // Cannot revisit a visited node
    if (visited[v->id]) {
        return;
    }

    // Create a queue for BFS
    queue<int> q;

    visited[v->id] = true;
    q.push(v->id);

    unsigned int seed = (unsigned int) time(NULL);
 
    // Using queue to do BFS, adding activated neighbors to the queue and pop the parent
    while(!q.empty())
    {
        int size = q.size();
 
        for (int i = 0; i < size; i++) {
            int cur = q.front();
            for (int j = 0; j < int(g->vertices[cur]->neighbors.size()); j++) {
                // Decide if to visit the neighbor according to propagation probability
                bool isVisit = (rand_r(&seed) / (float) RAND_MAX) <= g->prob;
                if (!isVisit) {
                    continue;
                }

                // Try to visit the neighbor
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

/**
 * Serial version of Monte Carlo simulation
 * Given the seed node set, run numIterations rounds simulation on the graph, and return the average result
 * The larger numIteration will give more precise simulation result
 */
int monteCarloSimulation(Graph *g, vector<int> vertices, int numIterations) {
    long result = 0;
    int nVertices = int(g->vertices.size());
    
    // Perform numIterations rounds of simulation
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

/**
 * Serial version of Monte Carlo simulation using pointer vertices instead of a vector
 * Given the seed node set, run numIterations rounds simulation on the graph, and return the average result
 * Try to use this method to optimize performance but it did not work
 */
int monteCarloSimulationPt(Graph *g, int* vertices, int verticesSize, int numIterations) {
    long result = 0;
    int nVertices = int(g->vertices.size());
    
    // Perform numIterations rounds of simulation
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

/**
 * Serial version of algorithm implementations, greedy and heuristic
 */
void compute(char *inputFilename, int nSeeds, int nMonteCarloSimulations, double prob, bool greedy, int heuristicMode) {
    /* initialize random seed: */
    srand(time(NULL));
    
    // Read input file to get the number of vertices, number of edges and all the information of the graph
    int nVertices = 0, nEdges = 0;
    Graph *g = new Graph(prob);
    readInput(inputFilename, &nVertices, &nEdges, &g);

    if (greedy) {
        // Greedy algorithm
        int maxval = INT_MIN;
        vector<int> seeds, selectedVertices;
        vector<bool> vec(nVertices, false);
        fill(vec.begin(), vec.begin() + nSeeds, true);

        /*
         * Generate a possible permutation of seed node set each time
         * Test the result using Monte Carlo simulation
         * Choose the seed node set with the largest Monte Carlo simulation result
         */
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
            // Basic heuristic algorithm, just sort all vertices by its out-degree in descending order, select top N
            vector<pair<int, int> > rank;
            for (int i = 0; i < nVertices; i++) {
                pair<int, int> p = make_pair(i, int(g->vertices[i]->neighbors.size()));
                rank.push_back(p);
            }
            sort(rank.begin(), rank.end(), cmp);
            
            // Select top N vertices as seed node set
            vector<int> seeds;
            for (int i = 0; i < nSeeds; i++) {
                seeds.push_back(rank[i].first);
            }
            // Test spread result
            int result = monteCarloSimulation(g, seeds, nMonteCarloSimulations);
            printf("Basic heuristic result = %d\n", result);
        } else if (mode == MINUSONE) {
            /*
             * Minus-one heuristic algorithm
             * Every time select the vertex with highest out-degree, and minus all its neighbors out-degree by 1
             */
            unordered_map<int, int> id2degree;
            for (int i = 0; i < nVertices; i++) {
                id2degree[g->vertices[i]->id] = int(g->vertices[i]->neighbors.size());
            }

            vector<int> seeds;
            for (int i = 0; i < nSeeds; i++) {
                int maxId = -1;
                int maxDegree = INT_MIN;
                
                // Find the vertex with highest out-degree and select it as seed node
                for (auto it = id2degree.begin(); it != id2degree.end(); it++) {
                    if (it->second > maxDegree) {
                        maxId = it->first;
                        maxDegree = it->second;
                    }
                }
                seeds.push_back(maxId);
                id2degree.erase(maxId);
                
                // Minus all its neighbors degree by 1
                for (int j = 0; j < int(g->vertices[maxId]->neighbors.size()); j++) {
                    id2degree[g->vertices[maxId]->neighbors[j]] -= 1;
                }
            }
            int result = monteCarloSimulation(g, seeds, nMonteCarloSimulations);
            printf("Minus-1 heuristic result = %d\n", result);
        } else if (mode == DEGREEDISCOUNT) {
            /*
             * Degree-Discount heuristic algorithm
             * Every time select the vertex with highest out-degree, and minus all its neighbors out-degree by a formula
             * ddv = dv - [2tv + (dv - tv)tvp]
             * Where dv = its original degree, tv = # of its neighbors as seed nodes, p = propagation probability
             */
            unordered_map<int, int> id2degree;
            for (int i = 0; i < nVertices; i++) {
                id2degree[g->vertices[i]->id] = int(g->vertices[i]->neighbors.size());
            }

            unordered_set<int> seedsSet;            
            for (int i = 0; i < nSeeds; i++) {
                int maxId = -1;
                int maxDegree = INT_MIN;

                // Find the vertex with highest out-degree and select it as seed node
                for (auto it = id2degree.begin(); it != id2degree.end(); it++) {
                    if (it->second > maxDegree) {
                        maxId = it->first;
                        maxDegree = it->second;
                    }
                }
                seedsSet.insert(maxId);
                id2degree.erase(maxId);
                
                // ddv = dv - [2tv + (dv - tv)tvp]
                // Where dv = its original degree, tv = # of its neighbors as seed nodes, p = propagation probability
                for (int j = 0; j < int(g->vertices[maxId]->neighbors.size()); j++) {
                    if (id2degree.find(g->vertices[maxId]->neighbors[j]) == id2degree.end()) continue;
                    int neighborId = g->vertices[maxId]->neighbors[j];
                    
                    // dv
                    int originalDegree =  int(g->vertices[neighborId]->neighbors.size());
                    
                    // tv
                    int neighborSeedsCnt = 0;
                    for (int k = 0; k < int(g->vertices[neighborId]->neighbors.size()); k++) {
                        if (seedsSet.find(g->vertices[neighborId]->neighbors[k]) != seedsSet.end()) {
                            neighborSeedsCnt++;
                        }
                    }

                    // Update degree using formula
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

/* 
 * Parallel implementation of BFS given a node and visited memo
 * Note that the paralellism is not inside this function, this function is called in parallel
 * The idea is, parallelize the BFS process of a set of nodes
 * Because different nodes can try to visit another node simultaneously, and a node can only be visited once
 * We have to use lock to protect the visited memo
 */
void singleNodeBFSParallelWithLock(Graph *g, int v_id, bool visited[], omp_lock_t *locks, int nVertices) {
    Vertex *v = g->vertices[v_id];
    // Create a queue for BFS
    queue<int> q;
    
    // Before checking if current node is visited, lock 
    omp_set_lock(&locks[v->id]);
    if (visited[v->id]) {
        omp_unset_lock(&locks[v->id]);
        return;
    }
    
    // Not visited, set visited as true, unlock
    visited[v->id] = true;
    omp_unset_lock(&locks[v->id]);

    q.push(v->id);

    unsigned int seed = (unsigned int) time(NULL);
 
    // Using queue to do BFS, adding activated neighbors to the queue and pop the parent
    while(!q.empty())
    {
        int size = q.size();
 
        for (int i = 0; i < size; i++) {
            int cur = q.front();
            for (int j = 0; j < int(g->vertices[cur]->neighbors.size()); j++) {
                // Decide if to visit the neighbor according to propagation probability
                bool isVisit = (rand_r(&seed) / (float) RAND_MAX) <= g->prob;
                if (!isVisit) {
                    continue;
                }

                int visit_id = g->vertices[cur]->neighbors[j];

                // Try to visit the neighbor
                // Before checking if current node is visited, lock 
                omp_set_lock(&locks[visit_id]);
                if (visited[visit_id]) {
                    omp_unset_lock(&locks[visit_id]);
                    continue;
                }

                // Not visited, set visited as true, unlock
                visited[visit_id] = true;
                omp_unset_lock(&locks[visit_id]);

                q.push(visit_id);
            }
            q.pop();
        }
    }
}

/**
 * Parallel version of Monte Carlo Simulation with parallelism in BFS process, not in rounds of simulations.
 * Acquire the corresponding lock before updating the visit array.
 */
int monteCarloSimulationParallelWithLock(Graph *g, vector<int> vertices, int numIterations, int nThreads) {
    long result = 0;
    int nVertices = int(g->vertices.size());
    
    for (int round = 0; round < numIterations; round++) {
        bool visited[nVertices];
        // locks for each vertex
        omp_lock_t *locks = (omp_lock_t *)calloc(nVertices, sizeof(omp_lock_t));
        
        // Init locks
        for (int i = 0; i < nVertices; i++) {
            visited[i] = false;
            omp_init_lock(&locks[i]);
        }

        // Parallel the BFS process of each vertex in the given vertices
        // Each vertex search parallelly, they synchronize using locks
        #pragma omp parallel for shared(locks)
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

/**
 * Parallel the monte carlo simulation in each round
 * Each thread take responsibility of one round independently for every nThreads round
 * Using reduction to sum all results
 */
int monteCarloSimulationParallel(Graph *g, vector<int> vertices, int numIterations, int nThreads) {
    long result = 0;
    int nVertices = int(g->vertices.size());
    
    // Sum up each final result of propagation
    #pragma omp parallel for reduction(+ : result)
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

/**
 * Parallel version of algorithm implementations, greedy and heuristic
 */
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
        // Greedy algorithm
        int maxval = INT_MIN;
        vector<int> seeds, selectedVertices;
        vector<bool> vec(nVertices, false);
        fill(vec.begin(), vec.begin() + nSeeds, true);

        /*
         * To parallelize the greedy algorithm, we tried two ways
         * The first is to parallelize the computation of Monte Carlo simulation 
         * for each possible permutation of seed nodes,
         * The second is to first compute all the possible permutations, 
         * then parallelize the computation of all permutations.
         * The second has better performance, so here we use the second
         */

        // First get the total number of all possible permutations
        int numPermutations = 0;
        do {
            numPermutations++;
        } while (prev_permutation(vec.begin(), vec.end()));

        /*
         * We have to precompute all the possible permutations
         * The required memory is large, so every time we compute 1/GREEDY_DIVIDE of it
         * We only need to allocate 1/GREEDY_DIVIDE memory
         * We do our parallel computation on 1/GREEDY_DIVIDE of all permutations
         */

        // Compute how many memory we have to exactly allocate because of padding problem
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

            // Get the current 1/GREEDY_DIVIDE of all permutations, write into allPermutations
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
            
            // Parallelly calculate each permutation's spread result which distrubuted in several blocks
            int i, nThreads;
            #pragma omp parallel default(shared) private(i, nThreads) 
            {
                i = omp_get_thread_num();
                nThreads = omp_get_num_threads();
                for (i = 0; i < curSize; i += nThreads) {
                    // vector<int> perm(allPermutations[i], allPermutations[i] + nSeeds);
                    // int cur = monteCarloSimulation(g, perm, nMonteCarloSimulations);
                    
                    int cur = monteCarloSimulationPt(g, allPermutations[i], nSeeds, nMonteCarloSimulations);
                    
                    // Each thread will update the max value, so we have to protect the critical section 
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
        // The global max spread result
        printf("maxval = %d\n", maxval);

        // free memory
        for(int pt = 0; pt < lineSize; pt++) {
            delete []allPermutations[pt];
        }
        delete []allPermutations;
    } else {
        // Heuristic
        int mode = heuristicMode;
        if (mode == BASIC) {
            // Basic heuristic algorithm, just sort all vertices by its out-degree in descending order, select top N
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
            
            // Choose to use which level of parallelism
            if (withLock) {
                // Parallel in BFS level, 
                result = monteCarloSimulationParallelWithLock(g, seeds, nMonteCarloSimulations, nThreads);
            } else {
                // Parallel in monte carlo simulation rounds level
                result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            }
            printf("Basic heuristic result = %d\n", result);
        } else if (mode == MINUSONE) {
            /*
             * Minus-one heuristic algorithm
             * Every time select the vertex with highest out-degree, and minus all its neighbors out-degree by 1
             */
            unordered_map<int, int> id2degree;
            for (int i = 0; i < nVertices; i++) {
                id2degree[g->vertices[i]->id] = int(g->vertices[i]->neighbors.size());
            }

            vector<int> seeds;
            for (int i = 0; i < nSeeds; i++) {
                int maxId = -1;
                int maxDegree = INT_MIN;
                // Find the vertex with highest out-degree and select it as seed node
                for (auto it = id2degree.begin(); it != id2degree.end(); it++) {
                    if (it->second > maxDegree) {
                        maxId = it->first;
                        maxDegree = it->second;
                    }
                }
                seeds.push_back(maxId);
                id2degree.erase(maxId);

                // Minus all its neighbors degree by 1
                for (int j = 0; j < int(g->vertices[maxId]->neighbors.size()); j++) {
                    id2degree[g->vertices[maxId]->neighbors[j]] -= 1;
                }
            }

            int result = 0;
            // Choose to use which level of parallelism
            if (withLock) {
                // Parallel in BFS level, 
                result = monteCarloSimulationParallelWithLock(g, seeds, nMonteCarloSimulations, nThreads);
            } else {
                // Parallel in monte carlo simulation rounds level
                result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            }
            printf("Minus-1 heuristic result = %d\n", result);
        } else if (mode == DEGREEDISCOUNT) {
            /*
             * Degree-Discount heuristic algorithm
             * Every time select the vertex with highest out-degree, and minus all its neighbors out-degree by a formula
             * ddv = dv - [2tv + (dv - tv)tvp]
             * Where dv = its original degree, tv = # of its neighbors as seed nodes, p = propagation probability
             */
            unordered_map<int, int> id2degree;
            // #pragma omp parallel for
            for (int i = 0; i < nVertices; i++) {
                id2degree[g->vertices[i]->id] = int(g->vertices[i]->neighbors.size());
            }

            unordered_set<int> seedsSet;            
            for (int i = 0; i < nSeeds; i++) {
                int maxId = -1;
                int maxDegree = INT_MIN;
                // Find the vertex with highest out-degree and select it as seed node
                for (auto it = id2degree.begin(); it != id2degree.end(); it++) {
                    if (it->second > maxDegree) {
                        maxId = it->first;
                        maxDegree = it->second;
                    }
                }
                seedsSet.insert(maxId);
                id2degree.erase(maxId);
                
                // ddv = dv - [2tv + (dv - tv)tvp]
                // Where dv = its original degree, tv = # of its neighbors as seed nodes, p = propagation probability
                for (int j = 0; j < int(g->vertices[maxId]->neighbors.size()); j++) {
                    if (id2degree.find(g->vertices[maxId]->neighbors[j]) == id2degree.end()) continue;
                    int neighborId = g->vertices[maxId]->neighbors[j];

                    // dv
                    int originalDegree =  int(g->vertices[neighborId]->neighbors.size());
                    
                    // tv
                    int neighborSeedsCnt = 0;
                    for (int k = 0; k < int(g->vertices[neighborId]->neighbors.size()); k++) {
                        if (seedsSet.find(g->vertices[neighborId]->neighbors[k]) != seedsSet.end()) {
                            neighborSeedsCnt++;
                        }
                    }

                    // Update degree using the formula
                    id2degree[g->vertices[maxId]->neighbors[j]] = 
                        originalDegree - (2 * neighborSeedsCnt + 
                        (originalDegree - neighborSeedsCnt) * neighborSeedsCnt * g->prob);
                }
            }

            vector<int> seeds;
            seeds.insert(seeds.end(), seedsSet.begin(), seedsSet.end());

            int result = 0;
            // Choose to use which level of parallelism
            if (withLock) {
                // Parallel in BFS level, 
                result = monteCarloSimulationParallelWithLock(g, seeds, nMonteCarloSimulations, nThreads);
            } else {
                // Parallel in monte carlo simulation rounds level
                result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            }
            printf("Degree Discount heuristic result = %d\n", result);
        } else if (mode == DEGREEDISCOUNTPARALLEL) {
            /*
             * Parallel Degree-Discount heuristic algorithm
             *
             * Serial version: Every time select the vertex with highest out-degree, and minus all its neighbors out-degree by a formula
             * ddv = dv - [2tv + (dv - tv)tvp]
             * Where dv = its original degree, tv = # of its neighbors as seed nodes, p = propagation probability
             *
             * Parallel idea:
             * Because selecting a node with highest degree will change all its neighbors' degrees, they have dependencies
             * To parallelize this, if we want to have S seed nodes and T threads, the idea is like "take blocks every time"
             * Imagine we have a data structure that map vertex id to its degree, and the map is automatically sorted descendingly by its value.
             * Suppose the data structure can support directly get its key value pair with specific rank
             * Then we can take T vertices each time and update their neighbors
             * There is a tradeoff between spread result and speed, because we can parallellize this process to make it run faster,
             * but we may select worse seed node set because every time we choose T vertices, the effect of degree-discount is ignored between them
             *
             * This can be parallelized in three steps:
             * 1. Get top T vertices parallely and add them to seed nodes set, delete them from id2degree
             * 2. Update every selected seed nodes' neighbors' degree by formula in parallel, update in id2degree
             * 3. Repeat step 1 and 2 until select S seed nodes
             * 
             * However, there is no such data structure in C++. We have found another way to do this
             * The data structures we use, is a int-int map id2degree and int-vector<int> map degree2ids
             * id2degree holds every nodes' degree, degree2ids holds every degree's corresponding vertex ids
             * Because C++'s map is sorted by its key, degree2ids can let us get vertices with highest degree, but not parallelizable
             * 
             * The step is as follows:
             * 1. Get top T vertices in serial and add them to seed nodes set, delete them from id2degree and degree2ids
             * 2. Update every selected seed nodes' neighbors' degree by formula in parallel, update in id2degree and degree2ids
             * 3. Repeat step 1 and 2 until select S seed nodes
             * The parallelism in getting top T vertices cannot be implemented, so the parallelism is only in step 2
             * Note that we have to use locks in id2degree and degree2ids to prevent contention because updating degree is parallel
             */

            unordered_map<int, pair<omp_lock_t*, int>> id2degree;
            // #pragma omp parallel for shared(id2degree)  // No effect
            for (int i = 0; i < nVertices; i++) {
                // To prevent contention, we assign a lock for each vertex
                // Lock is needed to be acquired before operating the vertex's degree
                omp_lock_t* lock = new omp_lock_t();
                omp_init_lock(lock);
                pair<omp_lock_t*, int> newVertice = {lock, int(g->vertices[i]->neighbors.size())};
                id2degree[g->vertices[i]->id] = newVertice;
            }

            // This map is sorted by descending value of degree, key is a degree and value is a set of all vertex ids with this degree
            // Because we cannot call omp_init_lock() in #pragma omp to avoid segmentation fault,
            // we have to initialize all the possible locks before entering parallel section
            map<int, pair<omp_lock_t*, unordered_set<int>>, greater<int> > degree2ids;
            for (int i = 0; i <= g->vertices.size(); i++) {
                unordered_set<int> uset;
                
                // To prevent contention, we assign a lock for each set
                // Lock is needed to be acquired before operating the set
                omp_lock_t* lock = new omp_lock_t();
                omp_init_lock(lock);
                pair<omp_lock_t*, unordered_set<int>> newVertice = {lock, uset};

                degree2ids[i] = newVertice; 
            }

            // Initialize all nodes' degrees
            for (auto it = id2degree.begin(); it != id2degree.end(); it++) {
                degree2ids[it->second.second].second.insert(it->first);
            }
                
            // Store total selected seeds
            unordered_set<int> seedsSet;  
            int totalCount = 0;

            // Fetch nThreads seeds every time
            for (int i = 0; i < nSeeds; i += nThreads) {
                int count = 0;
                // Store current round selected seeds
                vector<int> tempSeeds;
                // get # nThreads and use them to update neighbors' degree in parallel
                for (auto it1 = degree2ids.begin(); it1 != degree2ids.end(); it1++) {
                    unordered_set<int> uset = degree2ids[it1->first].second;
                    // Store selected seeds in current uset, record for deletion purpose
                    vector<int> innerSeeds;
                    for (auto it2 = uset.begin(); it2 != uset.end(); it2++) {
                        int id = *it2;
                        seedsSet.insert(id);
                        tempSeeds.push_back(id);
                        innerSeeds.push_back(id);
                        id2degree.erase(id);
                        count++;
                        totalCount++;
                        // Read enough seeds in this round or in total
                        if (count == nThreads || totalCount == nSeeds) break;
                    }
                    // Delect the selected seeds in current uset
                    for (int ii = 0; ii < innerSeeds.size(); ii++) {
                        degree2ids[it1->first].second.erase(innerSeeds[ii]);
                    }
                    if (count == nThreads || totalCount == nSeeds) break;
                }
                
                /*
                * Degree-Discount heuristic algorithm
                * Every time select the vertex with highest out-degree, and minus all its neighbors out-degree by a formula
                * ddv = dv - [2tv + (dv - tv)tvp]
                * Where dv = its original degree, tv = # of its neighbors as seed nodes, p = propagation probability
                */
                
                // Parallel computing of degree-discount for all seed's neighbors selected in this round
                #pragma omp parallel for
                for (int currSeed = 0; currSeed < tempSeeds.size(); currSeed++) {
                    Vertex* v = g->vertices[currSeed];
                    for (int j = 0; j < int(v->neighbors.size()); j++) {
                        int neighborId = v->neighbors[j];
                        // Neighbor has been selected as seed node
                        if (id2degree.find(neighborId) == id2degree.end()) continue;

                        // dv
                        int originalDegree = int(g->vertices[neighborId]->neighbors.size());

                        // tv
                        int neighborSeedsCnt = 0;
                        for (int k = 0; k < int(g->vertices[neighborId]->neighbors.size()); k++) {
                            if (seedsSet.find(g->vertices[neighborId]->neighbors[k]) != seedsSet.end()) {
                                neighborSeedsCnt++;
                            }
                        }

                        // ddv = dv - [2tv + (dv - tv)tvp]
                        int updateTo = originalDegree - (2 * neighborSeedsCnt + 
                            (originalDegree - neighborSeedsCnt) * neighborSeedsCnt * g->prob);
                            
                        // Trade off, in order to fit in range, we have to sacrifice the precision
                        // The reason is we cannot call omp_lock_init() inside #pragma omp to avoid segmentation fault
                        if (updateTo < 0) {
                            updateTo = 0;
                        }

                        // update id2degree
                        int liveDegree;
                        // Accuire lock first
                        omp_set_lock(id2degree[neighborId].first);
                        liveDegree = id2degree[neighborId].second;
                        // Modify degree
                        id2degree[neighborId].second = updateTo;
                        // Release lock
                        omp_unset_lock(id2degree[neighborId].first);

                        // update degree2ids
                        // Get original degree and delete it
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
            // Choose to use which level of parallelism
            if (withLock) {
                // Parallel in BFS level, 
                result = monteCarloSimulationParallelWithLock(g, seeds, nMonteCarloSimulations, nThreads);
            } else {
                // Parallel in monte carlo simulation rounds level
                result = monteCarloSimulationParallel(g, seeds, nMonteCarloSimulations, nThreads);
            }
            printf("Parallel Degree Discount Heuristic result = %d\n", result);
        }
    }
}