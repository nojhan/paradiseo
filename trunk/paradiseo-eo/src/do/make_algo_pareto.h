// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_algo_pareto.h
// (c) Maarten Keijzer, Marc Schoenauer and GeNeura Team, 2002
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

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@inria.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _make_algo_pareto_h
#define _make_algo_pareto_h

#include "utils/eoData.h"     // for eo_is_a_rate
// everything tha's needed for the algorithms - SCALAR fitness

// Selection
// the eoSelectOne's
#include "eoSelectFromWorth.h"
#include "eoNDSorting.h"

// Breeders
#include "eoGeneralBreeder.h"

// Replacement - at the moment limited to eoNDPlusReplacement, locally defined
#include "eoReplacement.h"

template <class EOT, class WorthT = double>
class eoNDPlusReplacement : public eoReplacement<EOT>
{
public:
  eoNDPlusReplacement(eoPerf2Worth<EOT, WorthT>& _perf2worth) : perf2worth(_perf2worth) {}

  struct WorthPair : public std::pair<WorthT, const EOT*>
  {
    bool operator<(const WorthPair& other) const { return other.first < this->first; }
  };

  void operator()(eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
  {
    unsigned sz = _parents.size();
    _parents.reserve(_parents.size() + _offspring.size());
    copy(_offspring.begin(), _offspring.end(), back_inserter(_parents));

    // calculate worths
    perf2worth(_parents);
    perf2worth.sort_pop(_parents);
    perf2worth.resize(_parents, sz);

    _offspring.clear();
  }

private :
  eoPerf2Worth<EOT, WorthT>& perf2worth;
};


// Algorithm (only this one needed)
#include "eoEasyEA.h"

  // also need the parser and param includes
#include "utils/eoParser.h"
#include "utils/eoState.h"


/*
 * This function builds the algorithm (i.e. selection and replacement)
 *      from existing continue (or checkpoint) and operators
 *
 * It uses a parser (to get user parameters) and a state (to store the memory)
 * the last argument is an individual, needed for 2 reasons
 *     it disambiguates the call after instanciations
 *     some operator might need some private information about the indis
 *
 * This is why the template is the complete EOT even though only the fitness
 * is actually templatized here
*/

template <class EOT>
eoAlgo<EOT> & do_make_algo_pareto(eoParser& _parser, eoState& _state, eoEvalFunc<EOT>& _eval, eoContinue<EOT>& _continue, eoGenOp<EOT>& _op)
{
  // the selection
  std::string & selStr = _parser.createParam(std::string("NSGA-II"), "selCrit", "Pareto Selection Criterion: NSGA, NSGA-II, ParetoRanking", 'S', "Evolution Engine").value();
  double nicheSize = _parser.createParam(1.0, "nicheSize", "Size of niche for NSGA-I", '\0', "Evolution Engine").value();
  eoPerf2Worth<EOT, double> *p2w;
  if ( (selStr == std::string("NSGA")) || (selStr == std::string("NSGA-I") ) )
    p2w = new eoNDSorting_I<EOT>(nicheSize);
  else   if (selStr == std::string("NSGA-II"))
    p2w = new eoNDSorting_II<EOT>();
  else   if (selStr == std::string("ParetoRanking"))
    {
      eoDominanceMap<EOT>&  dominance = _state.storeFunctor(new eoDominanceMap<EOT>);
    p2w = new eoParetoRanking<EOT>(dominance);
    }

  _state.storeFunctor(p2w);

  // now the selector (from p2w) - cut-and-pasted from make_algo_scalar!
  // only all classes are now ...FromWorth ...
  // only the ranking is not re-implemented (yet?)
  eoValueParam<eoParamParamType>& selectionParam = _parser.createParam(eoParamParamType("DetTour(2)"), "selection", "Selection: Roulette, DetTour(T), StochTour(t) or Random", 'S', "Evolution Engine");

  eoParamParamType & ppSelect = selectionParam.value(); // std::pair<std::string,std::vector<std::string> >

  eoSelectOne<EOT>* select ;
  if (ppSelect.first == std::string("DetTour")) 
  {
    unsigned detSize;

    if (!ppSelect.second.size())   // no parameter added
      {
	std::cerr << "WARNING, no parameter passed to DetTour, using 2" << std::endl;
	detSize = 2;
	// put back 2 in parameter for consistency (and status file)
	ppSelect.second.push_back(std::string("2"));
      }
    else	  // parameter passed by user as DetTour(T)
      detSize = atoi(ppSelect.second[0].c_str());
    select = new eoDetTournamentWorthSelect<EOT>(*p2w, detSize);
  }
  else if (ppSelect.first == std::string("StochTour"))
    {
      double p;
      if (!ppSelect.second.size())   // no parameter added
	{
	  std::cerr << "WARNING, no parameter passed to StochTour, using 1" << std::endl;
	  p = 1;
	  // put back p in parameter for consistency (and status file)
	  ppSelect.second.push_back(std::string("1"));
	}
      else	  // parameter passed by user as DetTour(T)
	p = atof(ppSelect.second[0].c_str());
      
      select = new eoStochTournamentWorthSelect<EOT>(*p2w, p);
    }
//   else if (ppSelect.first == std::string("Sequential")) // one after the other
//     {
//       bool b;
//       if (ppSelect.second.size() == 0)   // no argument -> default = ordered
// 	{
// 	  b=true;
// 	  // put back in parameter for consistency (and status file)
// 	  ppSelect.second.push_back(std::string("ordered"));
// 	}
//       else
// 	b = !(ppSelect.second[0] == std::string("unordered"));
//       select = new eoSequentialWorthSelect<EOT>(b);
//     }
  else if (ppSelect.first == std::string("Roulette")) // no argument (yet)
    {
      select = new eoRouletteWorthSelect<EOT>(*p2w);
    }
  else if (ppSelect.first == std::string("Random")) // no argument, no perf2Worth
    {
      select = new eoRandomSelect<EOT>;
    }
  else
    {
      std::string stmp = std::string("Invalid selection: ") + ppSelect.first;
      throw std::runtime_error(stmp.c_str());
    }

  _state.storeFunctor(select);


  // the number of offspring 
    eoValueParam<eoHowMany>& offspringRateParam =  _parser.createParam(eoHowMany(1.0), "nbOffspring", "Nb of offspring (percentage or absolute)", 'O', "Evolution Engine");

  // the replacement
    // actually limited to eoNDPlusReplacement
  eoReplacement<EOT> & replace = _state.storeFunctor(
           new eoNDPlusReplacement<EOT, double>(*p2w)
	   );

  // the general breeder
  eoGeneralBreeder<EOT> *breed = 
    new eoGeneralBreeder<EOT>(*select, _op, offspringRateParam.value());
  _state.storeFunctor(breed);

  // now the eoEasyEA
  eoAlgo<EOT> *algo = new eoEasyEA<EOT>(_continue, _eval, *breed, replace);
  _state.storeFunctor(algo);
  // that's it!
  return *algo;
}

#endif
