#include <stdio.h>
#include <stdlib.h>

int main() {

    int* ptr = NULL;
    printf("1. Initial ptr value: %p\n", (void*)ptr); // 1. Initial ptr value: (nil)

    if (ptr == NULL) {
        printf("2. ptr is NULL, cannot dereference\n"); // 2. ptr is NULL, cannot dereference
    }

    ptr = malloc(sizeof(int));
    if (ptr == NULL) {
        printf("3. Memory allocation failed\n");
        return 1;
    }

    printf("4. After allocation, ptr value: %p\n", (void*)ptr); // 4. After allocation, ptr value: 0x55c309c8c760
    printf("4.1 After allocation, ptr value: %p\n", ptr); // 4.1 After allocation, ptr value: 0x55ae60a03760 --> while not necessary for all machines but better to type cast to void*

    *ptr = 42;
    printf("5. Value at ptr: %d\n", *ptr); // 5. Value at ptr: 42

    free(ptr);

    printf("5.1 After free but before setting it to NULL, ptr value: %p\n", (void*)ptr); // 5.1 After free but before setting it to NULL, ptr value: 0x55ae60a03760
    printf("5.2 After free but before setting it to NULL, Value at ptr: %d\n", *ptr); // 5.2 After free but before setting it to NULL, Value at ptr: 1476779617

    ptr = NULL;  // Set to NULL after freeing

    printf("6. After free and setting it to NULL, ptr value: %p\n", (void*)ptr); // 6. After free and setting it to NULL, ptr value: (nil)
    printf("6.1 After free and setting it to NULL, ptr value: %p\n", ptr); // 6.1 After free and setting it to NULL, ptr value: (nil)

    if (ptr == NULL) {
        printf("7. ptr is NULL, safely avoided use after free\n"); // 7. ptr is NULL, safely avoided use after free
    }

    return 0;
}
