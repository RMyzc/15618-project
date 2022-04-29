#include "influence.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <unistd.h>
#include <utility>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;

void printHelp(char *argv[]) {
    printf("******************************************************\n");
    printf("* Parallel Influence Maximization in Social Networks *\n");
    printf("*          Author: Zican Yang, Yuling Wu             *\n");
    printf("******************************************************\n");
    printf("Parameters:\n");
    printf("\t<-f FILE_PATH> Test file path\n");
    printf("\t[-p PROBABILITY] Propagation probability, default 0.1\n");
    printf("\t[-i MONTE_CARLO_SIMULATIONS] Iterations for monte carlo simulation, default 100\n");
    printf("\t[-s # of SEEDS] Number of initial seeds, default 10\n");
    printf("\t[-t IS_GREEDY] Use greedy method, default 1\n");
    printf("\t[-m PARALLEL_MODE] Use parallel mode, default 1\n");
    printf("\t[-n # of THREADS] Number of threads, default 1\n");
    printf("\t[-x HEURISTIC_MODE] Heuristic approach, 0 for BASIC, 1 for MINUS_ONE, 2 for DEGREE_DISCOUNT, default 2\n");
    printf("\t[-l WITH_LOCK] Lock implemantation in heuristic approach, default 0\n");
    printf("\t[-h]: Print helper\n");
    printf("\n");
    printf("Usage: \n\t%s -f <filename> [-p <PROBABILITY>] [-i <MONTE_CARLO_SIMULATIONS>]"
            "[-s <# SEEDS>] [-t <IS_GREEDY>]"
            "[-m <PARALLEL_MODE>] [-n <# THREADS>]"
            "[-x <HEURISTIC_MODE>] [-l <WITH_LOCK>] [-h]\n", argv[0]);
    printf("\n");
}

int main(int argc, char *argv[]) {
    double totalRuntime;
    int nSeeds = 10;
    int nMonteCarloSimulations = 100;
    double prob = 0.1;
    char *inputFilename = NULL;
    int opt = 0;
    bool greedy = true;
    int mode = true;
    int nthreads = 1;
    int heuristicMode = DEGREEDISCOUNT;
    bool withLock = false;

    // Read command line arguments
    do {
        opt = getopt(argc, argv, "f:p:i:s:t:m:n:x:l:h");
        switch (opt) {
        case 'f':
            inputFilename = optarg;
            break;

        case 'p':
            prob = atof(optarg);
            break;

        case 'i':
            nMonteCarloSimulations = atoi(optarg);
            break;

        case 's':
            nSeeds = atoi(optarg);
            break;

        case 't':
            greedy = atoi(optarg) > 0;
            break;

        case 'm':
            mode = atoi(optarg) > 0;
            break;

        case 'n':
            nthreads = atoi(optarg);
            break;

        case 'x':
            heuristicMode = atoi(optarg);
            break;
        
        case 'l':
            withLock = atoi(optarg) > 0;
            break;
        
        case 'h':
            printHelp(argv);
            break;
            
        case -1:
            break;

        default:
            break;
        }
    } while (opt != -1);

    if (inputFilename == NULL) {
        printf("No input file detected, please use `-h` for help.\n");
        return -1;
    }

    auto startTime = Clock::now();
    // Run computation
    if (mode) {
        compute(inputFilename, nSeeds, nMonteCarloSimulations, prob, greedy, heuristicMode);
    } else {
        computeParallel(inputFilename, nSeeds, nMonteCarloSimulations, prob, greedy, nthreads, heuristicMode, withLock);
    }
    
    totalRuntime = chrono::duration_cast<dsec>(Clock::now() - startTime).count();
    printf("Total Compute Time: %lf.\n", totalRuntime);

    // Cleanup
    // printf("Elapsed time for proc %d: %f\n", procID, endTime - startTime);
    return 0;
}