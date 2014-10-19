// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_algo_easea.h
// (c) Marc Schoenauer and Pierre Collet, 2002
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

    Contact: Pierre.Collet@polytechnique.fr
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _make_algo_easea_h
#define _make_algo_easea_h

#include <utils/eoData.h>     // for eo_is_a_rate
// everything tha's needed for the algorithms - SCALAR fitness

// Selection
// the eoSelectOne's
#include <eoRandomSelect.h>
#include <eoSequentialSelect.h>
#include <eoDetTournamentSelect.h>
#include <eoProportionalSelect.h>
#include <eoFitnessScalingSelect.h>
#include <eoRankingSelect.h>
#include <eoStochTournamentSelect.h>
// #include <eoSelect.h>    included in all others

// Breeders
#include <eoGeneralBreeder.h>

// Replacement
#include "make_general_replacement.h"
#include "eoMGGReplacement.h"
#include "eoG3Replacement.h"


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
 *
 *
 * @ingroup Builders
*/
template <class EOT>
eoAlgo<EOT> & do_make_algo_scalar(eoParser& _parser, eoState& _state, eoPopEvalFunc<EOT>& _popeval, eoContinue<EOT>& _continue, eoGenOp<EOT>& _op)
{
  // the selection
  eoValueParam<eoParamParamType>& selectionParam = _parser.createParam(eoParamParamType("DetTour(2)"), "selection", "Selection: Roulette, Ranking(p,e), DetTour(T), StochTour(t), Sequential(ordered/unordered) or EliteSequentialSelect", 'S', "Evolution Engine");

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
    else          // parameter passed by user as DetTour(T)
      detSize = atoi(ppSelect.second[0].c_str());
    select = new eoDetTournamentSelect<EOT>(detSize);
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
      else        // parameter passed by user as DetTour(T)
        p = atof(ppSelect.second[0].c_str());

      select = new eoStochTournamentSelect<EOT>(p);
    }
  else if (ppSelect.first == std::string("Ranking"))
    {
      double p,e;
      if (ppSelect.second.size()==2)   // 2 parameters: pressure and exponent
        {
          p = atof(ppSelect.second[0].c_str());
          e = atof(ppSelect.second[1].c_str());
        }
      else if (ppSelect.second.size()==1)   // 1 parameter: pressure
        {
          std::cerr << "WARNING, no exponent to Ranking, using 1" << std::endl;
          e = 1;
          ppSelect.second.push_back(std::string("1"));
          p = atof(ppSelect.second[0].c_str());
        }
      else // no parameters ... or garbage
        {
          std::cerr << "WARNING, no parameter to Ranking, using (2,1)" << std::endl;
          p=2;
          e=1;
          // put back in parameter for consistency (and status file)
          ppSelect.second.resize(2); // just in case
          ppSelect.second[0] = (std::string("2"));
          ppSelect.second[1] = (std::string("1"));
        }
      // check for authorized values
      // pressure in (0,1]
      if ( (p<=1) || (p>2) )
        {
          std::cerr << "WARNING, selective pressure must be in (1,2] in Ranking, using 2\n";
          p=2;
          ppSelect.second[0] = (std::string("2"));
        }
      // exponent >0
      if (e<=0)
        {
          std::cerr << "WARNING, exponent must be positive in Ranking, using 1\n";
          e=1;
          ppSelect.second[1] = (std::string("1"));
        }
      // now we're OK
      eoPerf2Worth<EOT> & p2w = _state.storeFunctor( new eoRanking<EOT>(p,e) );
      select = new eoRouletteWorthSelect<EOT>(p2w);
    }
  else if (ppSelect.first == std::string("Sequential")) // one after the other
    {
      bool b;
      if (ppSelect.second.size() == 0)   // no argument -> default = ordered
        {
          b=true;
          // put back in parameter for consistency (and status file)
          ppSelect.second.push_back(std::string("ordered"));
        }
      else
        b = !(ppSelect.second[0] == std::string("unordered"));
      select = new eoSequentialSelect<EOT>(b);
    }
  else if (ppSelect.first == std::string("EliteSequential")) // Best first, one after the other in random order afterwards
    {
      select = new eoEliteSequentialSelect<EOT>;
    }
  else if (ppSelect.first == std::string("Roulette")) // no argument (yet)
    {
      select = new eoProportionalSelect<EOT>;
    }
  else if (ppSelect.first == std::string("Random")) // no argument
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

  /////////////////////////////////////////////////////
  // the replacement
  /////////////////////////////////////////////////////

  /** Replacement type - high level: predefined replacements
   * ESComma :
   *          elite = 0
   *          surviveParents=0 (no reduce)
   *          surviveOffspring=100% (no reduce)
   *          reduceFinal = Deterministic
   *
   * ESPlus : idem, except for
   *          surviveParents = 100%
   *
   * GGA : generational GA - idem ESComma except for
   *          offspringRate = 100%
   *          all reducers are unused
   *
   * SSGA(T/t) : Steady-State GA
   *               surviveParents = 1.0 - offspringRate
   *               reduceFinal = DetTour(T>1) ou StochTour(0.5<t<1)
   *
   * EP(T) : Evolutionary Programming
   *           offspringRate=100%
   *           surviveParents = 100%
   *           surviveOffspring = 100%
   *           reduceFinal = EP(T)
   *
   * G3 and MGG are excetions at the moment, treated on their own
   *
   */

  eoParamParamType & replacementParam = _parser.createParam(eoParamParamType("General"), "replacement", "Type of replacement: General, or Generational, ESComma, ESPlus, SSGA(T), EP(T), G3, MGG(T)", '\0', "Evolution Engine").value();
  // the pointer
  eoReplacement<EOT> * ptReplace;

  // first, separate G3 and MGG
  // maybe one day we have a common class - but is it really necessary???
  if (replacementParam.first == std::string("G3"))
    {
    // reduce the parents: by default, survive parents = -2 === 2 parents die
    eoHowMany surviveParents =  _parser.createParam(eoHowMany(-2,false), "surviveParents", "Nb of surviving parents (percentage or absolute)", '\0', "Evolution Engine / Replacement").value();
    // at the moment, this is the only argument
    ptReplace = new eoG3Replacement<EOT>(-surviveParents);    // must receive nb of eliminated parets!
    _state.storeFunctor(ptReplace);
    }
  else  if (replacementParam.first == std::string("MGG"))
    {
      float t;
      unsigned tSize;
    // reduce the parents: by default, survive parents = -2 === 2 parents die
    eoHowMany surviveParents =  _parser.createParam(eoHowMany(-2,false), "surviveParents", "Nb of surviving parents (percentage or absolute)", '\0', "Evolution Engine / Replacement").value();
    // the tournament size
    if (!replacementParam.second.size())   // no parameter added
      {
        std::cerr << "WARNING, no parameter passed to MGG replacement, using 2" << std::endl;
        tSize = 2;
        // put back 2 in parameter for consistency (and status file)
        replacementParam.second.push_back(std::string("2"));
      }
    else
      {
        t = atof(replacementParam.second[0].c_str());
        if (t>=2)
          {                        // build the appropriate deafult value
            tSize = unsigned(t);
          }
        else
          {
            throw std::runtime_error("Sorry, only deterministic tournament available at the moment");
          }
      }
    ptReplace = new eoMGGReplacement<EOT>(-surviveParents, tSize);
    _state.storeFunctor(ptReplace);
    }
  else {   // until the end of what was the only loop/switch

  // the default deafult values
  eoHowMany elite (0.0);
  bool strongElitism (false);
  eoHowMany surviveParents (0.0);
  eoParamParamType reduceParentType ("Deterministic");
  eoHowMany surviveOffspring (1.0);
  eoParamParamType reduceOffspringType ("Deterministic");
  eoParamParamType reduceFinalType ("Deterministic");

  // depending on the value entered by the user, change some of the above
  double t;

  // ---------- General
  if (replacementParam.first == std::string("General"))
  {
    ;                              // defaults OK
  }
  // ---------- ESComma
  else if (replacementParam.first == std::string("ESComma"))
  {
    ;                              // OK too
  }
  // ---------- ESPlus
  else if (replacementParam.first == std::string("ESPlus"))
  {
    surviveParents = eoHowMany(1.0);
  }
  // ---------- Generational
  else if (replacementParam.first == std::string("Generational"))
  {
    ;                        // OK too (we should check nb of offspring)
  }
  // ---------- EP
  else if (replacementParam.first == std::string("EP"))
  {
    if (!replacementParam.second.size())   // no parameter added
      {
        std::cerr << "WARNING, no parameter passed to EP replacement, using 6" << std::endl;
        // put back 6 in parameter for consistency (and status file)
        replacementParam.second.push_back(std::string("6"));
      }
    // by coincidence, the syntax for the EP reducer is the same than here:
    reduceFinalType = replacementParam;
    surviveParents = eoHowMany(1.0);
  }
  // ---------- SSGA
  else if (replacementParam.first == std::string("SSGA"))
  {
    if (!replacementParam.second.size())   // no parameter added
      {
        std::cerr << "WARNING, no parameter passed to SSGA replacement, using 2" << std::endl;
        // put back 2 in parameter for consistency (and status file)
        replacementParam.second.push_back(std::string("2"));
        reduceParentType = eoParamParamType(std::string("DetTour(2)"));
      }
    else
      {
        t = atof(replacementParam.second[0].c_str());
        if (t>=2)
          {                        // build the appropriate deafult value
            reduceParentType = eoParamParamType(std::string("DetTour(") + replacementParam.second[0].c_str() + ")");
          }
        else   // check for [0.5,1] will be made in make_general_replacement
          {                        // build the appropriate deafult value
            reduceParentType = eoParamParamType(std::string("StochTour(") + replacementParam.second[0].c_str() + ")");
          }
      }
    //
    surviveParents = eoHowMany(-1);
    surviveOffspring = eoHowMany(1);
  }
  else                 // no replacement recognized
    {
      throw std::runtime_error("Invalid replacement type " + replacementParam.first);
    }

  ptReplace = & make_general_replacement<EOT>(
     _parser, _state, elite, strongElitism, surviveParents, reduceParentType, surviveOffspring, reduceOffspringType, reduceFinalType);

  } // end of the ugly construct due to G3 and MGG - totaly heterogeneous at the moment


  ///////////////////////////////
  // the general breeder
  ///////////////////////////////
  eoGeneralBreeder<EOT> *breed =
    new eoGeneralBreeder<EOT>(*select, _op, offspringRateParam.value());
  _state.storeFunctor(breed);

  ///////////////////////////////
  // now the eoEasyEA
  ///////////////////////////////
  eoAlgo<EOT> *algo = new eoEasyEA<EOT>(_continue, _popeval, *breed, *ptReplace);
  _state.storeFunctor(algo);
  // that's it!
  return *algo;
}

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
eoAlgo<EOT> & do_make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<EOT>& _eval, eoContinue<EOT>& _continue, eoGenOp<EOT>& _op)
{
         do_make_algo_scalar( _parser, _state, *(new eoPopLoopEval<EOT>(_eval)), _continue, _op);
}



#endif
