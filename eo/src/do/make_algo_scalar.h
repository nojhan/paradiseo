// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_algo_scalar.h
// (c) Maarten Keijzer, Marc Schoenauer and GeNeura Team, 2001
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
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _make_run_h
#define _make_run_h

#include <utils/eoData.h>     // for eo_is_a_rate
// everything tha's needed for the algorithms - SCALAR fitness

// Selection
// the eoSelectOne's
#include <eoRandomSelect.h>	// also contains the eoSequentialSelect
#include <eoDetTournamentSelect.h>
#include <eoProportionalSelect.h>
#include <eoFitnessScalingSelect.h>
#include <eoRankingSelect.h>
#include <eoStochTournamentSelect.h>

// Breeders
#include <eoGeneralBreeder.h>

// Replacement
// #include <eoReplacement.h>
#include <eoMergeReduce.h>
#include <eoReduceMerge.h>
#include <eoSurviveAndDie.h>

// Algorithm (only this one needed)
#include <eoEasyEA.h>

  // also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


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
eoAlgo<EOT> & do_make_algo_scalar(eoParameterLoader& _parser, eoState& _state, eoEvalFunc<EOT>& _eval, eoContinue<EOT>& _ccontinue, eoGenOp<EOT>& _op)
{
  // the selection
  eoValueParam<eoParamParamType>& selectionParam = _parser.createParam(eoParamParamType("DetTour(2)"), "selection", "Selection: Roulette, DetTour(T), StochTour(t) or Sequential(ordered/unordered)", 'S', "Evolution Engine");

  eoParamParamType & ppSelect = selectionParam.value(); // pair<string,vector<string> >

  eoSelectOne<EOT>* select ;
  if (ppSelect.first == string("DetTour")) 
  {
    unsigned size;
    istrstream is(ppSelect.second[0].c_str());  // size of det tournament
    is >> size;
    select = new eoDetTournamentSelect<EOT>(size);
  }
  else if (ppSelect.first == string("StochTour"))
    {
      double p;
      istrstream is(ppSelect.second[0].c_str()); // proba of binary tournament
      is >> p;
      select = new eoStochTournamentSelect<EOT>(p);
    }
  else if (ppSelect.first == string("Sequential")) // one after the other
    {
      bool b;
      if (ppSelect.second.size() == 0)   // no argument -> default = ordered
	b=true;
      else
	b = !(ppSelect.second[0] == string("unordered"));
      select = new eoSequentialSelect<EOT>(b);
    }
  else if (ppSelect.first == string("Roulette")) // no argument (yet)
    {
      select = new eoProportionalSelect<EOT>;
    }
  else
    {
      string stmp = string("Invalid selection: ") + ppSelect.first;
      throw runtime_error(stmp.c_str());
    }

  _state.storeFunctor(select);

  // the number of offspring 
    eoValueParam<eoHowMany>& offspringRateParam =  _parser.createParam(eoHowMany(1.0), "nbOffspring", "Nb of offspring (percentage or absolute)", 'O', "Evolution Engine");

  // the replacement
  eoValueParam<eoParamParamType>& replacementParam = _parser.createParam(eoParamParamType("Comma"), "replacement", "Replacement: Comma, Plus or EPTour(T), SSGAWorst, SSGADet(T), SSGAStoch(t)", 'R', "Evolution Engine");

  eoParamParamType & ppReplace = replacementParam.value(); // pair<string,vector<string> >

  eoReplacement<EOT>* replace ;
  if (ppReplace.first == string("Comma")) // Comma == generational
  {
    replace = new eoCommaReplacement<EOT>;
  }
  else if (ppReplace.first == string("Plus"))
    {
      replace = new eoPlusReplacement<EOT>;
    }
  else if (ppReplace.first == string("EPTour"))
    {
      unsigned size;
      istrstream is(ppReplace.second[0].c_str()); // size of EP tournament
      is >> size;
      replace = new eoEPReplacement<EOT>(size);
    }
  else if (ppReplace.first == string("SSGAWorst"))
    {
      replace = new eoSSGAWorseReplacement<EOT>;
    }
  else if (ppReplace.first == string("SSGADet"))
    {
      unsigned size;
      istrstream is(ppReplace.second[0].c_str()); // size of Det. tournament
      is >> size;
      replace = new eoSSGADetTournamentReplacement<EOT>(size);
    }
  else if (ppReplace.first == string("SSGAStoch"))
    {
      double p;
      istrstream is(ppReplace.second[0].c_str()); // proba of binary tournament
      is >> p;
      replace = new eoSSGAStochTournamentReplacement<EOT>(p);
    }
  else
    {
      string stmp = string("Invalid replacement: ") + ppReplace.first;
      throw runtime_error(stmp.c_str());
    }

  _state.storeFunctor(replace);

  // adding weak elitism
  eoValueParam<bool>& weakElitismParam =  _parser.createParam(false, "weakElitism", "Old best parent replaces new worst offspring *if necessary*", 'w', "Evolution Engine");
  if (weakElitismParam.value())
    {
      eoReplacement<EOT> *replaceTmp = replace;
      replace = new eoWeakElitistReplacement<EOT>(*replaceTmp);
      _state.storeFunctor(replace);
    }      

  // the general breeder
  eoGeneralBreeder<EOT> *breed = 
    new eoGeneralBreeder<EOT>(*select, _op, offspringRateParam.value());
  _state.storeFunctor(breed);

  // now the eoEasyEA
  eoAlgo<EOT> *algo = new eoEasyEA<EOT>(_ccontinue, _eval, *breed, *replace);
  _state.storeFunctor(algo);
  // that's it!
  return *algo;
}

#endif
