#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <stdexcept>  // runtime_error 
#ifdef HAVE_SSTREAM
#include <sstream>
#else
#include <strstream>
#endif

#include <paradiseo.h>
#include <ga.h>

typedef eoBit<double> Indi;	// A bitstring with fitness double

#include "binary_value.h"

using namespace std;

void main_function(int argc, char **argv) {
  
  eoEvalFuncPtr <Indi, double, const vector <bool> & > eval (binary_value) ;
  
  eoListener <Indi> listen (argc, argv) ;
    
  eoEvaluator <Indi> evaluator ("Mars",
				listen,
				eval) ;
  
  // Runs 
  evaluator () ;
}

// A main that catches the exceptions

int main(int argc, char **argv)
{

    try
    {
        main_function(argc, argv);
    }
    catch(exception& e)
    {
        cout << "Exception: " << e.what() << '\n';
    }

    return 1;
}
