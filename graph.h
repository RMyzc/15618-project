/**
 * CMU 15-618 Parallel Computer Architecture and Programming
 * Parallel Influence Maximization in Social Networks
 * 
 * Author: Zican Yang(zicany), Yuling Wu(yulingw)
 */
 
#include <vector>

using namespace std;

/**
 * Social Network Graph
 * Contains all vertices in the network and the propagation probability.
 */
class Graph {
public:
    vector<Vertex*> vertices; // All the vertices in the graph
    double prob; // propagation probability

    Graph() {
        vertices.clear();
        prob = 0.1;
    }

    Graph(double p) {
        vertices.clear();
        prob = p;
    }
};