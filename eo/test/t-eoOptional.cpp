//-----------------------------------------------------------------------------
// t-eoOptional.cpp
//-----------------------------------------------------------------------------

#include "eoOptional.h"

//-----------------------------------------------------------------------------

typedef int T;

struct MyClass {
    MyClass(eoOptional<T> my_T = NULL)
    : default_T(42), actual_T(my_T.getOr(default_T))
    {
    	std::cout << "Value " << actual_T << " was used for construction" << std::endl;
    }
private:
    T default_T;
    T& actual_T;
};

int main(int ac, char** av)
{
    // Three ways of using MyClass:
    MyClass mc1;
    MyClass mc2(NULL);
    T t(666);
    MyClass mc3(t);
}

