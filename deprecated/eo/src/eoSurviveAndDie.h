/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoSurviveAndDie.h
   (c) Maarten Keijzer, Marc Schoenauer, GeNeura Team, 2000

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

#ifndef _eoSurviveAndDie_h
#define _eoSurviveAndDie_h


//-----------------------------------------------------------------------------
#include <eoPop.h>
#include <eoFunctor.h>
#include <eoMerge.h>
#include <eoReduce.h>
#include <utils/eoHowMany.h>
//-----------------------------------------------------------------------------

/**
eoSurviveAndDie: takes a population (first argument),
kills the ones that are to die,
puts the ones that are to survive into the second argument
removes them from the first pop argument

@class eoSurviveAndDie
@class eoDeterministicSurviveAndDie,
@class eoDeterministicSaDReplacement
*/

/** @addtogroup Replacors
 * @{
 */

/** eoSurviveAndDie
A pure abstract class, to store the howmany's
*/
template <class EOT>
class eoSurviveAndDie : public eoBF<eoPop<EOT> &, eoPop<EOT> &, void>
{
public:
    eoSurviveAndDie(double _survive, double _die, bool _interpret_as_rate = true):
        howmanySurvive(_survive, _interpret_as_rate),
        howmanyDie(_die, _interpret_as_rate)
    {}

protected:
    eoHowMany howmanySurvive;
    eoHowMany howmanyDie;

};

/** An instance (theonly one as of today, Dec. 20, 2000) of an
    eoSurviveAndDie, that does everything deterministically

    Used in eoDeterministicSaDReplacement.
*/
template <class EOT>
class eoDeterministicSurviveAndDie : public eoSurviveAndDie<EOT>
{
public:

    using eoSurviveAndDie< EOT >::howmanyDie;
    using eoSurviveAndDie< EOT >::howmanySurvive;

    /** constructor */
    eoDeterministicSurviveAndDie(double _survive, double _die, bool _interpret_as_rate = true)
        : eoSurviveAndDie< EOT >(_survive, _die, _interpret_as_rate)
    {}


    void operator()(eoPop<EOT> & _pop, eoPop<EOT> & _luckyGuys)
    {
        unsigned pSize = _pop.size();
        unsigned nbSurvive = howmanySurvive(pSize);
        // first, save the best into _luckyGuys
        if (nbSurvive)
            {
                _pop.nth_element(nbSurvive);
                // copy best
                _luckyGuys.resize(nbSurvive);
                std::copy(_pop.begin(), _pop.begin()+nbSurvive, _luckyGuys.begin());
                // erase them from pop
                _pop.erase(_pop.begin(), _pop.begin()+nbSurvive);
            }
        unsigned nbRemaining = _pop.size();

        // carefull, we can have a rate of 1 if we want to kill all remaining
        unsigned nbDie = std::min(howmanyDie(pSize), pSize-nbSurvive);
        if (nbDie > nbRemaining)
            throw std::logic_error("eoDeterministicSurviveAndDie: Too many to kill!\n");

        if (!nbDie)
          {
            return;
          }
        // else
        // kill the worse nbDie
        _pop.nth_element(nbRemaining-nbDie);
        _pop.resize(nbRemaining-nbDie);
    }

};

/**
eoDeterministicSaDReplacement: replacement strategy that is just, in sequence
  saves best and kill worse from parents
+ saves best and kill worse from offspring
+ merge remaining (neither save nor killed) parents and offspring
+ reduce that merged population
= returns reduced pop + best parents + best offspring

An obvious use is as strong elitist strategy,
   i.e. preserving best parents, and reducing
         (either offspring or parents+offspring)
*/
template <class EOT>
class eoDeterministicSaDReplacement : public eoReplacement<EOT>
{
public:
  /**  Constructor with reduce */
  eoDeterministicSaDReplacement(eoReduce<EOT>& _reduceGlobal,
                 double _surviveParents, double _dieParents=0,
                 double _surviveOffspring=0, double _dieOffspring=0,
                 bool _interpret_as_rate = true ) :
        reduceGlobal(_reduceGlobal),
        sAdParents(_surviveParents, _dieParents, _interpret_as_rate),
        sAdOffspring(_surviveOffspring, _dieOffspring, _interpret_as_rate)
    {}

  /**  Constructor with default truncate used as reduce */
    eoDeterministicSaDReplacement(
                 double _surviveParents, double _dieParents=0,
                 double _surviveOffspring=0, double _dieOffspring=0,
                 bool _interpret_as_rate = true ) :
        reduceGlobal(truncate),
        sAdParents(_surviveParents, _dieParents, _interpret_as_rate),
        sAdOffspring(_surviveOffspring, _dieOffspring, _interpret_as_rate)
    {}

    void operator()(eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
    {
        unsigned pSize = _parents.size(); // target number of individuals

        eoPop<EOT> luckyParents;       // to hold the absolute survivors
        sAdParents(_parents, luckyParents);

        eoPop<EOT> luckyOffspring;       // to hold the absolute survivors
        sAdOffspring(_offspring, luckyOffspring);

        unsigned survivorSize = luckyOffspring.size() + luckyParents.size();
        if (survivorSize > pSize)
            throw std::logic_error("eoGeneralReplacement: More survivors than parents!\n");

        plus(_parents, _offspring); // all that remain in _offspring

        reduceGlobal(_offspring, pSize - survivorSize);
        plus(luckyParents, _offspring);
        plus(luckyOffspring, _offspring);

        _parents.swap(_offspring);

    }

private :
  eoReduce<EOT>& reduceGlobal;
  eoDeterministicSurviveAndDie<EOT> sAdParents;
  eoDeterministicSurviveAndDie<EOT> sAdOffspring;
  // plus helper (could be replaced by operator+= ???)
  eoPlus<EOT> plus;
  // the default reduce: deterministic truncation
  eoTruncate<EOT> truncate;
};


/** @} */

#endif
