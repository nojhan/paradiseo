//-----------------------------------------------------------------------------
// t-eofitness.cpp
// (c) GeNeura Team 1998
//-----------------------------------------------------------------------------

#include <time.h>    // time
#include <stdlib.h>  // srand, rand
#include <iostream>  // cout

#include <eoScalarFitness.h>

using namespace std;

//-----------------------------------------------------------------------------

template <class Fitness>
int test_fitness(Fitness a, Fitness b)
{
//  srand(time(0));

//  Fitness a = aval; //static_cast<double>(rand()) / RAND_MAX; 
//  Fitness b = bval; //static_cast<double>(rand()) / RAND_MAX;

  cout.precision(2);
  
  unsigned repeat = 2;
  while (repeat--)
    {
      cout << "------------------------------------------------------" << endl;
      cout << "testing <    ";
      if (a < b)
	cout << a << " < " << b << "  is true" << endl;
      else
	cout << a << " < " << b << "  is false" <<endl;
      
      cout << "testing >    ";
      if (a > b)
	cout << a << " > " << b << "  is true" << endl;
      else
	cout << a << " > " << b << "  is false" <<endl;
      
      cout << "testing ==   ";
      if (a == b)
	cout << a << " == " << b << " is true" << endl;
      else
	cout << a << " == " << b << " is false" <<endl;
      
      cout << "testing !=   ";
      if (a != b)
	cout << a << " != " << b << " is true" << endl;
      else
	cout << a << " != " << b << " is false" <<endl;
      
      a = b;
    }
  return 1;
}

int main()
{
    cout << "Testing minimizing fitness with 1 and 2" << endl;
    cout << "------------------------------------------------------" << endl;

    eoMinimizingFitness a = 1;
    eoMinimizingFitness b = 2;

    test_fitness(a, b);

    cout << "Testing minimizing fitness with 2 and 1" << endl;
    cout << "------------------------------------------------------" << endl;

    test_fitness(b, a);

    cout << "Testing maximizing fitness with 1 and 2" << endl;
    cout << "------------------------------------------------------" << endl;
    
    eoMaximizingFitness a1 = 1;
    eoMaximizingFitness b1 = 2;

    test_fitness(a1,b1);
    
    cout << "Testing maximizing fitness with 2 and 1" << endl;
    cout << "------------------------------------------------------" << endl;
    
    test_fitness(b1,a1);

}

//-----------------------------------------------------------------------------

