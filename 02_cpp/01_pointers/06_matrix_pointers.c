#include <stdio.h>


int main() {
    int arr1[] = {1, 2, 3, 4};
    int arr2[] = {5, 6, 7, 8};

    int* ptr1 = arr1;
    int* ptr2 = arr2;
    int* matrix[] = {ptr1, ptr2};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%d ", *matrix[i]++);
        }
        printf("\n");
    }

    printf("\n");

    // Above code is equivalent to:
    int* matrix1[] = {ptr1, ptr2};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%d ", (matrix1[i][j]));
        }
        printf("\n");
    }

    // Just testing
    int number = 7;
    int* p = &number;
    printf("%d ", *p);
    printf("%d ", p[0]); // This is equivalent to *p
    printf("\n");
}

// Output:
// 1 2 3 4 
// 5 6 7 8 

// 1 2 3 4 
// 5 6 7 8 
// 7 7 
