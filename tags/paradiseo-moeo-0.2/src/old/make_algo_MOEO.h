// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_algo_MOEO.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef _make_algo_MOEO_h
#define _make_algo_MOEO_h

// the parser and parameter includes
#include "utils/eoParser.h"
#include "utils/eoState.h"
// selections
#include "eoNDSorting.h"
#include "old/moeoIBEA.h"
#include "old/moeoBinaryQualityIndicator.h"
#include "eoParetoRanking.h"
#include "moeoParetoSharing.h"
#include "eoSelectFromWorth.h"
#include "moeoSelectOneFromPopAndArch.h"
// replacements
#include "eoReplacement.h"
#include "moeoReplacement.h"
// breeders
#include "eoGeneralBreeder.h"
// the algorithm
#include "eoEasyEA.h"

/*
 * This function builds the algorithm (i.e. selection and replacement) from existing continue (or checkpoint) and operators
 * It uses a parser (to get user parameters) and a state (to store the memory)
 *
 * NB: this function is almost cut-and-pasted from EO/make_algo_pareto.h and integrates MOEO features
 */
template < class EOT >
  eoAlgo < EOT > &do_make_algo_MOEO (eoParser & _parser, eoState & _state,
				     eoEvalFunc < EOT > &_eval,
				     eoContinue < EOT > &_continue,
				     eoGenOp < EOT > &_op,
				     moeoArchive < EOT > &_arch)
{

  // the fitness of an EOT object
  typedef typename EOT::Fitness EOFitness;





  /* the selection criteria */
  string & selStr = _parser.createParam (string ("NSGA-II"), "selCrit",
					 "Multi-objective selection criterion: NSGA, NSGA-II, IBEA, ParetoRanking, ParetoSharing",
					 'S', "Evolution Engine").value ();
  double nicheSize = _parser.createParam (1.0, "nicheSize",
					  "Size of niche for NSGA-I or ParetoSharing",
					  'n',
					  "Evolution Engine").value ();
  double kappa =
    _parser.createParam (0.05, "kappa", "Scaling factor kappa for IBEA", 'k',
			 "Evolution Engine").value ();
  string & indStr =
    _parser.createParam (string ("Epsilon"), "indicator",
			 "Binary quality indicator for IBEA : Epsilon, Hypervolume",
			 'I', "Evolution Engine").value ();
  double rho = _parser.createParam (1.1, "rho",
				    "reference point for the hypervolume calculation (must not be smaller than 1)",
				    'r', "Evolution Engine").value ();
  // the eoPerf2Worth object
  eoPerf2Worth < EOT, double >*p2w;
  if ((selStr == string ("NSGA")) || (selStr == string ("NSGA-I")))	// NSGA-I
    p2w = new eoNDSorting_I < EOT > (nicheSize);
  else if (selStr == string ("NSGA-II"))	// NSGA-II
    p2w = new eoNDSorting_II < EOT > ();
  else if (selStr == string ("IBEA"))
    {				// IBEA
      // the binary quality indicator
      moeoBinaryQualityIndicator < EOFitness > *I;
      if (indStr == string ("Epsilon"))
	I = new moeoAdditiveBinaryEpsilonIndicator < EOFitness >;
      else if (indStr == string ("Hypervolume"))
	I = new moeoBinaryHypervolumeIndicator < EOFitness > (rho);
      else
	{
	  string stmp =
	    string ("Invalid binary quality indicator (for IBEA): ") + indStr;
	  throw std::runtime_error (stmp.c_str ());
	}
      p2w = new moeoIBEASorting < EOT > (I, kappa);
    }
  else if (selStr == string ("ParetoRanking"))
    {				// Pareto Ranking
      eoDominanceMap < EOT > &dominance =
	_state.storeFunctor (new eoDominanceMap < EOT >);
      p2w = new eoParetoRanking < EOT > (dominance);
    }
  else if (selStr == string ("ParetoSharing"))
    {				// Pareto Sharing    
      p2w = new moeoParetoSharing < EOT > (nicheSize);
    }
  else
    {
      string stmp = string ("Invalid Pareto selection criterion: ") + selStr;
      throw std::runtime_error (stmp.c_str ());
    }
  // store  
  _state.storeFunctor (p2w);





  /* the selector */
  eoValueParam < eoParamParamType > &selectionParam =
    _parser.createParam (eoParamParamType ("DetTour(2)"), "selection",
			 "Selection: Roulette, DetTour(T), StochTour(t) or Random",
			 's', "Evolution Engine");
  eoParamParamType & ppSelect = selectionParam.value ();	// pair< string , vector<string> >
  // the select object
  eoSelectOne < EOT > *select;
  if (ppSelect.first == string ("DetTour"))
    {				// DetTour
      unsigned detSize;
      if (!ppSelect.second.size ())
	{			// no parameter added       
	  cerr << "WARNING, no parameter passed to DetTour, using 2" << endl;
	  detSize = 2;
	  // put back 2 in parameter for consistency (and status file)
	  ppSelect.second.push_back (string ("2"));
	}
      else			// parameter passed by user as DetTour(T)
	detSize = atoi (ppSelect.second[0].c_str ());
      select = new eoDetTournamentWorthSelect < EOT > (*p2w, detSize);
    }
  else if (ppSelect.first == string ("StochTour"))
    {				// StochTour
      double p;
      if (!ppSelect.second.size ())
	{			// no parameter added       
	  cerr << "WARNING, no parameter passed to StochTour, using 1" <<
	    endl;
	  p = 1;
	  // put back p in parameter for consistency (and status file)
	  ppSelect.second.push_back (string ("1"));
	}
      else			// parameter passed by user as DetTour(T)
	p = atof (ppSelect.second[0].c_str ());
      select = new eoStochTournamentWorthSelect < EOT > (*p2w, p);
    }
  else if (ppSelect.first == string ("Roulette"))
    {				// Roulette
      select = new eoRouletteWorthSelect < EOT > (*p2w);
    }
  else if (ppSelect.first == string ("Random"))
    {				// Random
      select = new eoRandomSelect < EOT >;
    }
  else
    {
      string stmp = string ("Invalid selection: ") + ppSelect.first;
      throw std::runtime_error (stmp.c_str ());
    }
  // store  
  _state.storeFunctor (select);





  /* elitism */
  bool useElitism = _parser.createParam (false, "elitism",
					 "Use elitism in the selection process (individuals from the archive are randomly selected)",
					 'E', "Evolution Engine").value ();
  double ratioFromPop = _parser.createParam (0.8, "ratio",
					     "Ratio from the population for elitism (must not be greater than 1)",
					     '\0',
					     "Evolution Engine").value ();
  if (useElitism)
    {
      eoSelectOne < EOT > *selectPop = select;
      select =
	new moeoSelectOneFromPopAndArch < EOT > (*selectPop, _arch,
						 ratioFromPop);
      // store  
      _state.storeFunctor (select);
    }





  /* the number of offspring  */
  eoValueParam < eoHowMany > &offspringRateParam =
    _parser.createParam (eoHowMany (1.0), "nbOffspring",
			 "Nb of offspring (percentage or absolute)", 'O',
			 "Evolution Engine");





  /* the replacement */
  string & repStr =
    _parser.createParam (string ("Plus"), "replacement",
			 "Replacement: Plus, DistinctPlus or Generational",
			 'R', "Evolution Engine").value ();
  eoReplacement < EOT > *replace;
  if (repStr == string ("Plus"))	// Plus
    {
      replace = new moeoElitistReplacement < EOT, double >(*p2w);
    }
  else if (repStr == string ("DistinctPlus"))	// DistinctPlus
    {
      replace = new moeoDisctinctElitistReplacement < EOT, double >(*p2w);
    }
  else if (repStr == string ("Generational"))	// Generational
    {
      replace = new eoGenerationalReplacement < EOT >;
    }
  else
    {
      string stmp = string ("Invalid replacement: ") + repStr;
      throw std::runtime_error (stmp.c_str ());
    }
  // store
  _state.storeFunctor (replace);





  // the general breeder
  eoGeneralBreeder < EOT > *breed =
    new eoGeneralBreeder < EOT > (*select, _op, offspringRateParam.value ());
  _state.storeFunctor (breed);

  // the eoEasyEA
  eoAlgo < EOT > *algo =
    new eoEasyEA < EOT > (_continue, _eval, *breed, *replace);
  _state.storeFunctor (algo);
  // that's it!
  return *algo;
}

#endif
