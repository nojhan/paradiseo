//-----------------------------------------------------------------------------
// t-eoOptional.cpp
//-----------------------------------------------------------------------------

#include "eoOptional.h"

//-----------------------------------------------------------------------------

typedef T int;

struct MyClass {
    MyClass(eoOptional<T> my_T = NULL)
    : actual_T(my_T.getOr(default_T))
    { }
private:
    T default_T;
    T& actual_T;
};

int main(int ac, char** av)
{
    // Three ways of using MyClass:
    MyClass mc1;
    MyClass mc2(NULL);
    T t;
    MyClass mc3(t);
}
