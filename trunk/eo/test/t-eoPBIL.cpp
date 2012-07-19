// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// t-eoPBIL.cpp
// (c) Marc Schoenauer, 2001
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: Marc.Schoenauer@inria.fr
 */
//-----------------------------------------------------------------------------

/** test program for PBIL algorithm */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <eo>
#include <ga/make_ga.h>
#include "binary_value.h"
#include <apply.h>
#include <ga/eoPBILDistrib.h>
#include <ga/eoPBILOrg.h>
#include <ga/eoPBILAdditive.h>
#include <eoSimpleEDA.h>


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
  eoEvalFuncPtr<Indi, double> mainEval( binary_value<Indi>);
  eoEvalFuncCounter<Indi> eval(mainEval);

  // Construction of the distribution
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
  eoValueParam<bool>& plotDistribParam = parser.getORcreateParam(false, "plotDistrib",
								 "Plot Distribution", '\0',
								 "Output - Graphical");
    if (plotDistribParam.value())
      {
#ifdef HAVE_GNUPLOT
	unsigned frequency=1;		// frequency of plots updates
	eoGnuplot1DSnapshot *distribSnapshot = new eoGnuplot1DSnapshot(ptDirNameParam->value(),
								       frequency, "distrib");
	state.storeFunctor(distribSnapshot);
	// add the distribution (it is an eoValueParam<vector<double> >)
	distribSnapshot->add(distrib);
	// and of course add it to the checkpoint
	checkpoint.add(*distribSnapshot);
#endif
      }

  // the algorithm: EDA
    // don't know where else to put the population size!
  unsigned popSize = parser.getORcreateParam(unsigned(100), "popSize",
					     "Population Size", 'P', "Algorithm").value();
  eoSimpleEDA<Indi> eda(update, eval, popSize, checkpoint);

  ///// End of construction of the algorith
  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(parser);

  //// GO
  ///////

  eda(distrib); // run the eda

  std::cout << "Final Distribution\n";
  distrib.printOn(std::cout);
  std::cout << std::endl;

  // wait - for graphical output
    if (plotDistribParam.value())
      {
	string foo;
	cin >> foo;
      }
  }
  catch(std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
}
