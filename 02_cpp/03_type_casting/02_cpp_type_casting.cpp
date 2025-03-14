using namespace std;

#include <iostream>

int main(){
    
    //C++ style casts are checked by the compiler. C style casts aren't and can fail at runtime.
    // Another big benefit is that the 4 different C++ style casts express the intent of the programmer more clearly.

    // Static pointer used for implicit conversions between types
    cout << "Static pointer cast" << endl;
    double pi = 3.14159;
    int rounded_pi = static_cast<int>(pi);
    cout << rounded_pi << endl;
    // Why you would prefer static cast over c style cast:
    // char c = 10;       // 1 byte
    // int *p = (int*)&c; // 4 bytes - succeeds, however we are using a 4 byte pointer to point to a 1 byte location
    // *p = 5; // run-time - stack corruption; we have overidden 3 additional bytes to the original location.
    // cout << "Hola, ";
    // cout << *p << endl;
    // int *q = static_cast<int*>(&c); // compile-time error

    // Dynamic pointer - used for safe downcasting in inheritance hierarchies
    cout << "Dynamic pointer cast" << endl;
    class Base { virtual void foo() {} };
    class Derived : public Base { };

    Base* base_ptr = new Derived;
    Derived* derived_ptr = dynamic_cast<Derived*>(base_ptr);
    cout << base_ptr << " " << derived_ptr << endl;

    Base* base_ptr1 = new Base;
    Derived* derived_ptr1 = dynamic_cast<Derived*>(base_ptr1); // Doesn't fail at run time 
    cout << base_ptr1 << " "  << derived_ptr1 << endl; // Generates 0 as an output

    // Constant pointer is used to add or remove const (or volatile) qualifiers from a variable.
    cout << "Constant pointer cast" << endl;
    const int constant = 10;
    int* non_const_ptr = const_cast<int*>(&constant);
    cout << constant << " " << *non_const_ptr << endl;
    *non_const_ptr = 20; // Modifies the const variable (undefined behavior)
    cout << constant << " " << *non_const_ptr << endl;
    cout << &constant << " " << non_const_ptr << endl; // points to the same address but different value :-o. This is most likely the compiler optimization

    //Re interpret cast - can convert between unrelated types, like pointers to integers or vice versa.
    //Should be used with extreme caution.
    cout << "Re interpret cast 1" << endl;
    int num = 65;
    char* char_ptr = reinterpret_cast<char*>(&num);
    cout << num << " " << *char_ptr << endl;
    cout << &num << " " << (void *) char_ptr << endl;
    cout << sizeof(num) << " " << sizeof(*char_ptr) << endl;
    cout << num << " " << *char_ptr << endl;
    *char_ptr = 'B';
    cout << num << " " << *char_ptr << endl;

    cout << "Re interpret cast 2" << endl;
    char char_a = 'A';
    int* int_ptr = reinterpret_cast<int*>(&char_a);
    cout << char_a << " " << *int_ptr << endl;
    cout << (void *) &char_a << " " << int_ptr << endl;
    cout << sizeof(char_a) << " " << sizeof(*int_ptr) << endl;
    cout << char_a << " " << *int_ptr << endl;
    *int_ptr = 66;
    cout << char_a << " " << *int_ptr << endl;
}

// Output: 
// Static pointer cast
// 3
// Dynamic pointer cast
// 0x55bca59812c0 0x55bca59812c0
// 0x55bca59812e0 0
// Constant pointer cast
// 10 10
// 10 20
// 0x7ffc3019df1c 0x7ffc3019df1c
// Re interpret cast 1
// 65 A
// 0x7ffc3019df20 0x7ffc3019df20
// 4 1
// 65 A
// 66 B
// Re interpret cast 2
// A 5185
// 0x7ffc3019df1b 0x7ffc3019df1b
// 1 4
// A 5185
// B 66
