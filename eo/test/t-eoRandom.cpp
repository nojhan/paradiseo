//-----------------------------------------------------------------------------
// t-eouniform
//-----------------------------------------------------------------------------

#include <iostream>   // cout
#include <strstream>  // ostrstream, istrstream
#include <eoUniform.h>         // eoBin
#include <eoNormal.h>
#include <eoNegExp.h>

//-----------------------------------------------------------------------------

main() {
  eoNormal<float> n1(-2.5,3.5);
  eoNormal<double> n2(0.003, 0.0005 );
  eoNormal<unsigned long> n3( 10000000U, 10000U);
  eoNegExp<float> e1(3.5);
  eoNegExp<double> e2(0.003 );
  eoNegExp<long> e3( 10000U);
  cout << "n1\t\tn2\t\tn3\t\te1\t\te2\t\te3" << endl;
  for ( unsigned i = 0; i < 100; i ++) {
    cout << n1() << "\t" << n2() << "\t" << n3() << "\t" <<
      e1() << "\t" << e2() << "\t" << e3() << endl;
  }
    
}

//-----------------------------------------------------------------------------
