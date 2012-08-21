#include <cassert>
#include <vector>
#include <atomic>

#include <smp>

#include "smpTestClass.h"

using namespace std;
using namespace paradiseo::smp;

void f(std::atomic<int> &x)
{
    for(int i = 0; i < 100; i++) {
        cout << x << endl;
        x++;
    }
}

void g(std::atomic<int> &x)
{
    for(int i = 0; i < 100; i++)
        x--;
}

void foo()
{
    std::cout << "Foo" << std::endl;
    //std::this_thread::sleep_for(std::chrono::seconds(1));
}

int main(void)
{
    //---------------------------------------------------
    std::atomic<int> nb(0);
 
    Thread t1(&f,std::ref(nb));
    Thread t2(&g,std::ref(nb));

    t1.join();
    t2.join();

    assert(nb == 0); // Test atomic_barrier
    
    //--------------------------------------------------
    try
    {
        t1.start(foo);
    
        t1.join();
    }
    catch(exception& e)
    {
        cout << "Exception: " << e.what() << '\n';
    }
    
    return 0;
}


