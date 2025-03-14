using namespace std;

#include <iostream>

typedef struct {
    float x;
    float y;
} Point;

int main() {
    Point p = {1.1, 2.5};
    cout << "size of Point: " << sizeof(Point) << endl; // Output: 8 bytes = 4 bytes (float x) + 4 bytes (float y)   
}

// Output:
// size of Point: 8
// Note: Since this is C++ you should use the command: g++ -o 03_exec ./03_struct.cpp instead of gcc -o 03_exec ./03_struct.cpp (or use nvcc)
