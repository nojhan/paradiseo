#include <iostream>

#include <eo>
#include <ga/make_ga.h>
#include "binary_value.h"
#include <apply.h>
#include <ga/eoPBILDistrib.h>
#include <ga/eoPBILOrg.h>
#include <ga/eoPBILAdditive.h>
#include <eoSimpleDEA.h>

using namespace std;

typedef eoBit<double> Indi;

// instanciating the outside subroutine that creates the distribution
#include "ga/make_PBILdistrib.h"
eoPBILDistrib<Indi> & make_PBILdistrib(eoParser& _parser, eoState&_state, Indi _eo)
{
  return do_make_PBILdistrib(_parser, _state, _eo);
}

// instanciating the outside subroutine that creates the update rule
#include "ga/make_PBILupdate.h"
eoDistribUpdater<Indi> & make_PBILupdate(eoParser& _parser, eoState&_state, Indi _eo)
{
  return do_make_PBILupdate(_parser, _state, _eo);
}


int main(int argc, char* argv[])
{

  try
  {
  eoParser parser(argc, argv);  // for user-parameter reading

  eoState state;    // keeps all things allocated

  ///// FIRST, problem or representation dependent stuff
  //////////////////////////////////////////////////////

  // The evaluation fn - encapsulated into an eval counter for output 
  eoEvalFuncPtr<Indi, float> mainEval( binary_value<Indi>);
  eoEvalFuncCounter<Indi> eval(mainEval);

  // COnstruction of the distribution
  eoPBILDistrib<Indi> & distrib = make_PBILdistrib(parser, state, Indi());
  // and the update rule
  eoDistribUpdater<Indi> & update = make_PBILupdate(parser, state, Indi());

  //// Now the representation-independent things
  //////////////////////////////////////////////

  // stopping criteria
  eoContinue<Indi> & term = make_continue(parser, state, eval);
  // output
  eoCheckPoint<Indi> & checkpoint = make_checkpoint(parser, state, eval, term);

  // add a graphical output for the distribution
  // first, get the direname from the parser 
  //    it has been enetered in make_checkoint

  eoParam* ptParam = parser.getParamWithLongName(string("resDir"));
  eoValueParam<string>* ptDirNameParam = dynamic_cast<eoValueParam<string>*>(ptParam);
  if (!ptDirNameParam)	// not found
    throw runtime_error("Parameter resDir not found where it was supposed to be");

  // now create the snapshot monitor
    eoValueParam<bool>& plotDistribParam = parser.createParam(false, "plotDistrib", "Plot Distribution", '\0', "Output - Graphical");
    if (plotDistribParam.value())
      {
	unsigned frequency=1;		// frequency of plots updates
	eoGnuplot1DSnapshot *distribSnapshot = new eoGnuplot1DSnapshot(ptDirNameParam->value(), frequency, "distrib");
	state.storeFunctor(distribSnapshot);
	// add the distribution (it is an eoValueParam<vector<double> >)
	distribSnapshot->add(distrib);
	// and of course add it to the checkpoint
	checkpoint.add(*distribSnapshot);
      }

  // the algorithm: DEA
    // don't know where else to put the population size!
  unsigned popSize = parser.createParam(unsigned(100), "popSize", "Population Size", 'P', "Algorithm").value();
  eoSimpleDEA<Indi> dea(update, eval, popSize, checkpoint);

  ///// End of construction of the algorith
  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(parser);

  //// GO
  ///////

  dea(distrib); // run the dea

  cout << "Final Distribution\n";
  distrib.printOn(cout);
  cout << endl;

  // wait - for graphical output
    if (plotDistribParam.value())
      {
	string foo;
	cin >> foo;
      }
  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
}
