// See eoParserUtils.h

#include <iostream.h>
#include <eoParserUtils.h>
/// Reproducible random seed

//----------------------------------
void InitRandom( Parser & parser) {
//----------------------------------
  unsigned long _seed;
  try {
    _seed = parser.getUnsignedLong("-S", "--seed", "0", 
				   "Seed for Random number generator" );
  }
  catch (logic_error & e)
    {
      cout << e.what() << endl;
      parser.printHelp();
      exit(1);
    }

  if (_seed == 0) {		   // use clock to get a "random" seed
    _seed = (unsigned long)( time( 0 ) );
    ostrstream s;
    s << _seed;
    parser.setParamValue("--seed", s.str());	   // so it will be printed out in the status file, and canbe later re-used to re-run EXACTLY the same run
  }
  rng.reseed(_seed);

  return;
}

