#include "influence.h"
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

int main(int argc, char *argv[]) {
    double startTime;
    double endTime;
    int nSeeds = 10;
    int nMonteCarloSimulations = 100;
    double prob = 0.1;
    char *inputFilename = NULL;
    int opt = 0;

    // Read command line arguments
    do {
        opt = getopt(argc, argv, "f:p:i:s:");
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

    // Run computation
    compute(inputFilename, nSeeds, nMonteCarloSimulations, prob, &startTime, &endTime);

    // Cleanup
    // printf("Elapsed time for proc %d: %f\n", procID, endTime - startTime);
    return 0;
}