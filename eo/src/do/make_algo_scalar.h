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

#ifndef _make_algo_scalar_h
#define _make_algo_scalar_h

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
#include <eoSharingSelect.h>
#include <utils/eoDistance.h>

// Breeders
#include <eoGeneralBreeder.h>

// Replacement
// #include <eoReplacement.h>
#include <eoMergeReduce.h>
#include <eoReduceMerge.h>
#include <eoSurviveAndDie.h>

// distance
#include <utils/eoDistance.h>

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
 * @ingroup Builders
*/
template <class EOT>
eoAlgo<EOT> & do_make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<EOT>& _eval, eoContinue<EOT>& _continue, eoGenOp<EOT>& _op, eoDistance<EOT> * _dist = NULL)
{
  // the selection : help and comment depend on whether or not a distance is passed
  std::string comment;
  if (_dist == NULL)
    comment = "Selection: DetTour(T), StochTour(t), Roulette, Ranking(p,e) or Sequential(ordered/unordered)";
      else
        comment = "Selection: DetTour(T), StochTour(t), Roulette, Ranking(p,e), Sharing(sigma_share) or Sequential(ordered/unordered)";

  eoValueParam<eoParamParamType>& selectionParam = _parser.createParam(eoParamParamType("DetTour(2)"), "selection", comment, 'S', "Evolution Engine");

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
  else if (ppSelect.first == std::string("Sharing"))
  {
    double nicheSize;

    if (!ppSelect.second.size())   // no parameter added
      {
        std::cerr << "WARNING, no parameter passed to Sharing, using 0.5" << std::endl;
        nicheSize = 0.5;
        // put back 2 in parameter for consistency (and status file)
        ppSelect.second.push_back(std::string("0.5"));
      }
    else          // parameter passed by user as DetTour(T)
      nicheSize = atof(ppSelect.second[0].c_str());
    if (_dist == NULL)             // no distance
          throw std::runtime_error("You didn't specify a distance when calling make_algo_scalar and using sharing");
    select = new eoSharingSelect<EOT>(nicheSize, *_dist);
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
          std::cerr << "WARNING, selective pressure must be in (0,1] in Ranking, using 2\n";
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

  // the replacement
  eoValueParam<eoParamParamType>& replacementParam = _parser.createParam(eoParamParamType("Comma"), "replacement", "Replacement: Comma, Plus or EPTour(T), SSGAWorst, SSGADet(T), SSGAStoch(t)", 'R', "Evolution Engine");

  eoParamParamType & ppReplace = replacementParam.value(); // std::pair<std::string,std::vector<std::string> >

  eoReplacement<EOT>* replace ;
  if (ppReplace.first == std::string("Comma")) // Comma == generational
  {
    replace = new eoCommaReplacement<EOT>;
  }
  else if (ppReplace.first == std::string("Plus"))
    {
      replace = new eoPlusReplacement<EOT>;
    }
  else if (ppReplace.first == std::string("EPTour"))
    {
      unsigned detSize;

      if (!ppReplace.second.size())   // no parameter added
        {
          std::cerr << "WARNING, no parameter passed to EPTour, using 6" << std::endl;
          detSize = 6;
          // put back in parameter for consistency (and status file)
          ppReplace.second.push_back(std::string("6"));
        }
      else        // parameter passed by user as EPTour(T)
        detSize = atoi(ppSelect.second[0].c_str());

      replace = new eoEPReplacement<EOT>(detSize);
    }
  else if (ppReplace.first == std::string("SSGAWorst"))
    {
      replace = new eoSSGAWorseReplacement<EOT>;
    }
  else if (ppReplace.first == std::string("SSGADet"))
    {
      unsigned detSize;

      if (!ppReplace.second.size())   // no parameter added
        {
          std::cerr << "WARNING, no parameter passed to SSGADet, using 2" << std::endl;
          detSize = 2;
          // put back in parameter for consistency (and status file)
          ppReplace.second.push_back(std::string("2"));
        }
      else        // parameter passed by user as SSGADet(T)
        detSize = atoi(ppSelect.second[0].c_str());

      replace = new eoSSGADetTournamentReplacement<EOT>(detSize);
    }
  else if (ppReplace.first == std::string("SSGAStoch"))
    {
      double p;
      if (!ppReplace.second.size())   // no parameter added
        {
          std::cerr << "WARNING, no parameter passed to SSGAStoch, using 1" << std::endl;
          p = 1;
          // put back in parameter for consistency (and status file)
          ppReplace.second.push_back(std::string("1"));
        }
      else        // parameter passed by user as SSGADet(T)
        p = atof(ppSelect.second[0].c_str());

      replace = new eoSSGAStochTournamentReplacement<EOT>(p);
    }
  else
    {
      std::string stmp = std::string("Invalid replacement: ") + ppReplace.first;
      throw std::runtime_error(stmp.c_str());
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
  eoAlgo<EOT> *algo = new eoEasyEA<EOT>(_continue, _eval, *breed, *replace);
  _state.storeFunctor(algo);
  // that's it!
  return *algo;
}
/** @example t-eoGA.cpp
 */

#endif
