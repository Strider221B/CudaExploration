#include <stdio.h>

int main() {
    int arr[] = {12, 24, 36, 48, 60};

    int* ptr = arr; 

    printf("Position one: %d\t", *ptr); 
    printf("Position one: %p\n", ptr); 

    for (int i = 0; i < 5; i++) {
        printf("%d\t", *ptr);
        printf("%p\n", ptr);
        ptr++;
    }
}

// Output:
// Position one: 12        Position one: 0x7ffeeba015d0
// 12      0x7ffeeba015d0
// 24      0x7ffeeba015d4
// 36      0x7ffeeba015d8
// 48      0x7ffeeba015dc
// 60      0x7ffeeba015e0
