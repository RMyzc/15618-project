#include <vector>

using namespace std;

class Graph {
public:
    vector<Vertex*> vertices;
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