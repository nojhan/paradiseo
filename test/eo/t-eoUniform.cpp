//-----------------------------------------------------------------------------
// t-eouniform
//-----------------------------------------------------------------------------

#include <iostream>   // std::cout
#include <strstream>  // ostrstream, istrstream
#include <paradiseo/eo/eoUniform.h>         // eoBin

//-----------------------------------------------------------------------------

main() {
  eoUniform<float> u1(-2.5,3.5);
  eoUniform<double> u2(0.003, 0 );
  eoUniform<unsigned long> u3( 10000U, 10000000U);
  std::cout << "u1\t\tu2\t\tu3" << std::endl;
  for ( unsigned i = 0; i < 100; i ++) {
    std::cout << u1() << "\t" << u2() << "\t" << u3() << std::endl;
  }

}

//-----------------------------------------------------------------------------
