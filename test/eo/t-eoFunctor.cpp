#include <paradiseo/eo/eoInit.h>
#include <paradiseo/eo/eoCounter.h>

void f(eoInit<int>& func)
{
    int i;
    func(i);
}

class Tester : public eoInit<int>
{
public :
    void operator()(int& i)
    {
	i=1;
    }
};

#include <iostream>
#include <paradiseo/eo/eoFixedLength.h>
#include <paradiseo/eo/eoVariableLength.h>

using namespace std;

int main(void)
{
    Tester test;

    eoFunctorStore store;

    /// make a counter and store it in 'store'
    eoInit<int>& cntr = make_counter(functor_category(test), test, store);

    eoUnaryFunctorCounter<eoInit<int> > cntr2(test);

    f(cntr);
    f(cntr2);
    f(cntr2);
    f(test);

    typedef eoVariableLength<double, int> EoType;
    EoType eo;

    eo.push_back(1);
    eo.push_back(2);

    return 1;
}
