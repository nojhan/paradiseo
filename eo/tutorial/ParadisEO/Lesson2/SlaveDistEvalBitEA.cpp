#include <stdexcept>  // runtime_error 
#include <iostream>   // cout
#include <strstream>  // ostrstream, istrstream

#include <paradiseo>
#include <ga.h>

typedef eoBit<double> Indi;	// A bitstring with fitness double

#include "binary_value.h"

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
