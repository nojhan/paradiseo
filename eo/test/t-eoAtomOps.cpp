// Program to test several EO-ES features

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif 

#include <string>
#include <iostream>
#include <iterator>

using namespace std;

// Operators we are going to test
#include <eoAtomCreep.h>
#include <eoAtomBitFlip.h>
#include <eoAtomRandom.h>
#include <eoAtomMutation.h>

// Several EOs
#include <eoString.h>

main(int argc, char *argv[]) {
  eoString<float> aString("123456");
  eoAtomCreep<char> creeper;
  eoAtomMutation< eoString<float> > mutator( creeper, 0.5 );
  
  cout << "Before aString " << aString;
  mutator( aString);
  cout << " after mutator " << aString;
 
  return 0; // to avoid VC++ complaints
}

