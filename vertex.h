#include <vector>

using namespace std;

class Vertex {
public:
    int id;
    vector<int> neighbors;

    Vertex(int index) {
        id = index;
        neighbors.clear();
    }
};