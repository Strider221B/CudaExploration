#include <stdio.h>

int main() {
    int arr[] = {12, 24, 36, 48, 60};

    size_t size = sizeof(arr) / sizeof(arr[0]); // nothing special about size_t, it's just a typedef for unsigned long long int. In this e.g. you could have just mentioned int
    printf("Size of arr: %zu\n", size); // %zu -> long unsigned int
    printf("size of size_t: %zu\n", sizeof(size_t));
    printf("int size in bytes: %zu\n", sizeof(int));
    return 0;
}

// Output:
// Size of arr: 5
// size of size_t: 8
// int size in bytes: 4
