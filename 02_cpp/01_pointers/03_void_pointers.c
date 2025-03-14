#include <stdio.h>

int main() {
    int num = 10;
    float fnum = 3.142;
    void* vptr;

    vptr = &num;
    printf("Integer: %d\n", *(int*)vptr);
    // printf("Integer: %d\n", *vptr); This throws a compile time error -> error: invalid use of void expression

    vptr = &fnum;
    printf("Float: %.2f\n", *(float*)vptr);
}

// fun fact: malloc() returns a void pointer but we see it as a pointer to a specific data type after the cast (int*)malloc(4) or (float*)malloc(4) etc.