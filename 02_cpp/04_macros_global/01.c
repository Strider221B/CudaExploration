#include <stdio.h>

#define PI 3.14159
#define AREA(r) (PI * r * r)

#ifndef radius
#define radius 70
#endif

#if radius > 10
#define radius 10
#elif radius < 5
#define radius 5
#else
#define radius 7
#endif

int main() {
    printf("Area of circle with radius %d: %f\n", radius, AREA(radius));
    return 0;
}

// Output:
// ~/Git/CudaExploration/02_cpp/04_macros_global$ gcc -o 01_exec ./01.c
// ./01.c:11: warning: "radius" redefined
//    11 | #define radius 10
//       | 
// ./01.c:7: note: this is the location of the previous definition
//     7 | #define radius 70
//       | 
// ~/Git/CudaExploration/02_cpp/04_macros_global$ ./01_exec 
// Area of circle with radius 10: 314.159000
