/**
 * CMU 15-618 Parallel Computer Architecture and Programming
 * Parallel Influence Maximization in Social Networks
 * 
 * Author: Zican Yang(zicany), Yuling Wu(yulingw)
 */

#include <vector>

using namespace std;

/**
 * A single vertex in the graph.
 */
class Vertex {
public:
    int id; // Unique id of the vertex
    vector<int> neighbors; // Neighbor ids of the vertex

    Vertex(int index) {
        id = index;
        neighbors.clear();
    }
};