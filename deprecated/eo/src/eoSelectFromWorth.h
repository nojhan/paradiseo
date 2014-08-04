/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoSelectFromWorth.h
   (c) Maarten Keijzer, Marc Schoenauer, 2001

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

    Contact: Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoSelectFromWorth_h
#define _eoSelectFromWorth_h

#include <iostream>
//-----------------------------------------------------------------------------
#include <eoSelectOne.h>
#include <eoPerf2Worth.h>
#include <utils/selectors.h>
//-----------------------------------------------------------------------------

/** selects one element from a population (is an eoSelectOne)
but the selection is based on a std::vector of Worth that is different
from the fitnesses (e.g. EO fitness is what Koza terms "raw fitness",
Worth is what the selection is based upon).

see class eoPerf2Worth: an eoStat that transforms fitnesses into Worthes

Note: Worthes will not always be doubles - see some multi-objective
techniques where it is a std::pair of doubles ...

It has to have a < operator it you want to call an existing
selector (see selector.h) - but of course you can write the whole
thing ...

@ingroup Selectors
*/
template <class EOT, class WorthType = double>
class eoSelectFromWorth : public eoSelectOne<EOT>
{
public:

    /* Default ctor from an eoPerf2Worth object */
    eoSelectFromWorth(eoPerf2Worth<EOT, WorthType>& _perf2Worth)
        : perf2Worth(_perf2Worth)
        {}

    /* setup the worthes */
    virtual void setup(const eoPop<EOT>& pop) {
        perf2Worth(pop);
#ifndef NDEBUG
        fitness.resize(pop.size());
        for (unsigned i = 0; i < pop.size(); ++i) {
            fitness[i] = pop[i].fitness();
        }
#endif
    }


protected:

    eoPerf2Worth<EOT, WorthType>& perf2Worth;

#ifndef NDEBUG
    std::vector<typename EOT::Fitness> fitness;
    void check_sync(unsigned index, const EOT& _eo) {
        if (fitness[index] != _eo.fitness()) {
            throw std::runtime_error("eoSelectFromWorth: fitnesses are not in sync");
        }
    }
#endif
};


/** An instance of eoSelectPerf2Worth that does selection from the Worthes
 *  using a ... determinisitic tournament, yes!

@ingroup Selectors
 */
template <class EOT, class WorthT = double>
class eoDetTournamentWorthSelect : public eoSelectFromWorth<EOT, WorthT>
{
public:

    using eoSelectFromWorth<EOT, WorthT>::perf2Worth;

    typedef typename std::vector<WorthT>::iterator worthIterator;

    /* Default ctor from an eoPerf2Worth object +  tournament size */
    eoDetTournamentWorthSelect(eoPerf2Worth<EOT, WorthT>& perf2Worth,
                               unsigned _tSize)
        : eoSelectFromWorth<EOT, WorthT>(perf2Worth), tSize(_tSize) {}

    /* Perform deterministic tournament on worthes by calling the
       appropriate fn see selectors.h */
    virtual const EOT& operator()(const eoPop<EOT>& pop) {
        worthIterator it = deterministic_tournament(perf2Worth.value().begin(),
                                                    perf2Worth.value().end(),
                                                    tSize);
        unsigned index = it - perf2Worth.value().begin();
#ifndef NDEBUG
        // check whether the stuff is still in sync
        check_sync(index, pop[index]);
#endif
        return pop[index];
    }


private:

    unsigned tSize;
};

/** An instance of eoSelectPerf2Worth that does selection from the Worthes
 *  using a ... stochastic tournament, yes!

@ingroup Selectors
 */
template <class EOT, class WorthT = double>
class eoStochTournamentWorthSelect : public eoSelectFromWorth<EOT, WorthT>
{
public:

    using eoSelectFromWorth<EOT, WorthT>::perf2Worth;

    typedef typename std::vector<WorthT>::iterator worthIterator;

    /* Default ctor from an eoPerf2Worth object +  tournament rate */
    eoStochTournamentWorthSelect(eoPerf2Worth<EOT, WorthT> &_perf2Worth, double _tRate)
        : eoSelectFromWorth<EOT, WorthT>(_perf2Worth), tRate(_tRate)
        {}

  /* Perform stochastic tournament on worthes
     by calling the appropriate fn in selectors.h
  */
  virtual const EOT& operator()(const eoPop<EOT>& _pop) {
      worthIterator it = stochastic_tournament(perf2Worth.value().begin(),
                                               perf2Worth.value().end(),
                                               tRate);

    unsigned index = it - perf2Worth.value().begin();

#ifndef NDEBUG
  // check whether the stuff is still in sync
  check_sync(index, _pop[index]);
#endif

    return _pop[index];
  }

private:
  double tRate;
};

/** An instance of eoSelectPerf2Worth that does selection from the Worthes
 *  using a ... roulette wheel selection, yes!

@ingroup Selectors
 */
template <class EOT, class WorthT = double>
class eoRouletteWorthSelect : public eoSelectFromWorth<EOT, WorthT>
{
public:

    using eoSelectFromWorth<EOT, WorthT>::perf2Worth;

    typedef typename std::vector<WorthT>::iterator worthIterator;


    /* Default ctor from an eoPerf2Worth object */
    eoRouletteWorthSelect(eoPerf2Worth<EOT, WorthT> &_perf2Worth)
        : eoSelectFromWorth<EOT, WorthT>(_perf2Worth)
        {}

  /* We have to override the default behavior to compute the total
   * only once!
   */
  virtual void setup(const eoPop<EOT>& _pop)
  {
    eoSelectFromWorth<EOT, WorthT>::setup(_pop);
    total = 0.0;
    for (worthIterator it = perf2Worth.value().begin();
         it<perf2Worth.value().end(); ++it)
      total += (*it);
  }

  /* Perform roulette wheel on worthes
     by calling the appropriate fn
     see selectors.h
  */
  virtual const EOT& operator()(const eoPop<EOT>& _pop) {
 //     cout << "On affiche les worths\n";
 //     for (unsigned i=0;
 //      i<perf2Worth.value().size();
 //      i++)
 //       cout << perf2Worth.value().operator[](i) << "\n";
 //     cout << endl;
      worthIterator it = roulette_wheel(perf2Worth.value().begin(),
                                        perf2Worth.value().end(),
                                        total);

   unsigned index = it - perf2Worth.value().begin();

#ifndef NDEBUG
  // check whether the stuff is still in sync
  this->check_sync(index, _pop[index]);
#endif

    return _pop[index];
    return _pop[it - perf2Worth.value().begin()];
  }

private:
  double total;
};

#endif
