#include <stdio.h>

typedef struct {
    float x;
    float y;
} Point;

int main() {
    Point p = {1.1, 2.5};
    printf("size of float: %zu\n", sizeof(float));
    printf("size of Point: %zu\n", sizeof(Point));  // Output: 8 bytes = 4 bytes (float x) + 4 bytes (float y)
    return 0;
}

// Output:
// size of float: 4
// size of Point: 8
