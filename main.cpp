#include "influence.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <unistd.h>
#include <utility>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;

int main(int argc, char *argv[]) {
    double totalRuntime;
    int nSeeds = 10;
    int nMonteCarloSimulations = 100;
    double prob = 0.1;
    char *inputFilename = NULL;
    int opt = 0;
    bool greedy = true;

    // Read command line arguments
    do {
        opt = getopt(argc, argv, "f:p:i:s:t:");
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

        case -1:
            break;

        default:
            break;
        }
    } while (opt != -1);

    if (inputFilename == NULL) {
        printf("Usage: %s -f <filename> [-p <P>] [-i <N_iters>]\n", argv[0]);
        return -1;
    }

    auto startTime = Clock::now();
    // Run computation
    compute(inputFilename, nSeeds, nMonteCarloSimulations, prob, greedy);

    totalRuntime = chrono::duration_cast<dsec>(Clock::now() - startTime).count();
    printf("Total Compute Time: %lf.\n", totalRuntime);

    // Cleanup
    // printf("Elapsed time for proc %d: %f\n", procID, endTime - startTime);
    return 0;
}