#include <stdio.h>

int main() {
    int value = 42;
    int* ptr1 = &value;
    int** ptr2 = &ptr1;
    int*** ptr3 = &ptr2;
    
    printf("Value: %p\n", ptr1);
    printf("Value: %p\n", ptr2);
    printf("Value: %p\n", ptr3);
    printf("Value: %d\n", ***ptr3);
}
