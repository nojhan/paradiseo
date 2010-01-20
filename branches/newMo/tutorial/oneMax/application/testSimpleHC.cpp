//-----------------------------------------------------------------------------
/** testSimpleHC.cpp
 *
 * SV - 12/01/10 
 *
 */
//-----------------------------------------------------------------------------

// standard includes
#define HAVE_SSTREAM

#include <stdexcept>  // runtime_error 
#include <iostream>   // cout
#include <sstream>  // ostrstream, istrstream
#include <fstream>
#include <string.h>

// the general include for eo
#include <eo>
#include <ga.h>

using namespace std;

//-----------------------------------------------------------------------------
// fitness function
#include <funcOneMax.h>
#include <eoInt.h>
#include <neighborhood/moBitNeighborhood.h>
#include <oneMaxBitNeighbor.h>

#include <eval/moFullEvalByModif.h>
#include <eval/moFullEvalByCopy.h>
#include <comparator/moNeighborComparator.h>
#include <continuator/moTrueContinuator.h>
#include <algo/moLocalSearch.h>
#include <explorer/moSimpleHCexplorer.h>

// REPRESENTATION
//-----------------------------------------------------------------------------
// define your individuals
typedef eoBit<unsigned> Indi;	
typedef moBitNeighbor<unsigned int> Neighbor ; // incremental evaluation
typedef moBitNeighborhood<Neighbor> Neighborhood ;

// GENERAL
//-----------------------------------------------------------------------------

void main_function(int argc, char **argv)
{
  /* =========================================================
   *
   * Parameters 
   *
   * ========================================================= */

  // First define a parser from the command-line arguments
  eoParser parser(argc, argv);

  // For each parameter, define Parameter, read it through the parser,
  // and assign the value to the variable

  eoValueParam<uint32_t> seedParam(time(0), "seed", "Random number seed", 'S');
  parser.processParam( seedParam );
  unsigned seed = seedParam.value();

  // description of genotype
  eoValueParam<unsigned int> vecSizeParam(8, "vecSize", "Genotype size", 'V');
  parser.processParam( vecSizeParam, "Representation" );
  unsigned vecSize = vecSizeParam.value();

  string fileOut("out.dat");
  eoValueParam<string> fileStatParam(fileOut.c_str(), "out", "A file to export results", 'o');
  parser.processParam( fileStatParam, "Persistence" );
  fileOut = fileStatParam.value();

  // the name of the "status" file where all actual parameter values will be saved
  string str_status = parser.ProgramName() + ".status"; // default value
  eoValueParam<string> statusParam(str_status.c_str(), "status", "Status file");
  parser.processParam( statusParam, "Persistence" );

  // do the following AFTER ALL PARAMETERS HAVE BEEN PROCESSED
  // i.e. in case you need parameters somewhere else, postpone these
  if (parser.userNeedsHelp())
    {
      parser.printHelp(cout);
      exit(1);
    }
  if (statusParam.value() != "")
    {
      ofstream os(statusParam.value().c_str());
      os << parser;		// and you can use that file as parameter file
    }


  /* =========================================================
   *
   * Random seed
   *
   * ========================================================= */

  //reproducible random seed: if you don't change SEED above, 
  // you'll aways get the same result, NOT a random run
  rng.reseed(seed);


  /* =========================================================
   *
   * Eval fitness function
   *
   * ========================================================= */

  FuncOneMax<Indi> eval(vecSize);

  moFullEvalByCopy<Neighbor > fulleval(eval);

  /* =========================================================
   *
   * Initilisation of the solution
   *
   * ========================================================= */

  // a Indi random initializer
  eoUniformGenerator<bool> uGen;
  eoInitFixedLength<Indi> random(vecSize, uGen);
      
  /* =========================================================
   *
   * evaluation of a neighbor solution
   *
   * ========================================================= */
  
  // no need if incremental evaluation with OneMaxBitNeighbor
//  Neighbor::setFullEvalFunc(eval);

  /* =========================================================
   *
   * the neighborhood of a solution
   *
   * ========================================================= */
  
  moNeighborComparator<Neighbor > comparator;

  Neighborhood neighborhood ;

  /* =========================================================
   *
   * a neighborhood explorator solution
   *
   * ========================================================= */
  
  moSimpleHCexplorer<Neighborhood> explorer(neighborhood, fulleval, comparator);

  /* =========================================================
   *
   * the local search algorithm
   *
   * ========================================================= */

  moTrueContinuator<Neighborhood> continuator;

  moLocalSearch< moSimpleHCexplorer<Neighborhood>, moTrueContinuator<Neighborhood> > localSearch(explorer, continuator);

  /* =========================================================
   *
   * execute the local search from random sollution
   *
   * ========================================================= */

  Indi solution;

  random(solution);

  eval(solution);

  std::cout << "initial: " << solution << std::endl ;

 localSearch(solution);
  
  std::cout << "final:   " << solution << std::endl ;

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
