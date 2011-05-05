// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoPBILAdditive.h
// (c) Marc Schoenauer, Maarten Keijzer, 2001
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

    Contact: Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoPBILAdditive_H
#define _eoPBILAdditive_H

#include <eoDistribUpdater.h>
#include <ga/eoPBILDistrib.h>

/**
 * Distribution Class for PBIL algorithm
 *      (Population-Based Incremental Learning, Baluja and Caruana 96)
 *
 * This class implements an extended update rule:
 * in the original paper, the authors used
 *
 *  p(i)(t+1) = (1-LR)*p(i)(t) + LR*best(i)
 *
 * here the same formula is applied, with some of the best individuals
 * and for some of the worst individuals (with different learning rates)
*/
template <class EOT>
class eoPBILAdditive :  public eoDistribUpdater<EOT>
{
public:
  /** Ctor with parameters
   *  using the default values is equivalent to using eoPBILOrg
   */
  eoPBILAdditive(double _LRBest, unsigned _nbBest = 1,
                double _tolerance=0.0,
                double _LRWorst = 0.0, unsigned _nbWorst = 0 ) :
    maxBound(1.0-_tolerance), minBound(_tolerance),
    LR(0.0), nbBest(_nbBest), nbWorst(_nbWorst)
  {
    if (nbBest+nbWorst == 0)
      throw std::runtime_error("Must update either from best or from worst in eoPBILAdditive");

    if (_nbBest)
      {
        lrb = _LRBest/_nbBest;
        LR += _LRBest;
      }
    else
      lrb=0.0;                     // just in case
    if (_nbWorst)
      {
        lrw = _LRWorst/_nbWorst;
        LR += _LRWorst;
      }
    else
      lrw=0.0;                     // just in case
  }

  /** Update the distribution from the current population */
  virtual void operator()(eoDistribution<EOT> & _distrib, eoPop<EOT>& _pop)
  {
    eoPBILDistrib<EOT>& distrib = dynamic_cast<eoPBILDistrib<EOT>&>(_distrib);

    std::vector<double> & p = distrib.value();

    unsigned i, popSize=_pop.size();
    std::vector<const EOT*> result;
    _pop.sort(result);    // is it necessary to sort the whole population?
                         // but I'm soooooooo lazy !!!

    for (unsigned g=0; g<distrib.size(); g++)
      {
        p[g] *= (1-LR);            // relaxation
        if (nbBest)                // update from some of the best
          for (i=0; i<nbBest; i++)
            {
              const EOT & best = (*result[i]);
              if ( best[g] )       // if 1, increase proba
                p[g] +=  lrb;
            }
        if (nbWorst)
          for (i=popSize-1; i>=popSize-nbWorst; i--)
            {
              const EOT & best = (*result[i]);
              if ( !best[g] )      // if 0, increase proba
                p[g] +=  lrw;
            }
        // stay in [0,1] (possibly strictly due to tolerance)
        p[g] = std::min(maxBound, p[g]);
        p[g] = std::max(minBound, p[g]);
      }
  }

private:
  double maxBound, minBound;    // proba stay away from 0 and 1 by at least tolerance
  double LR;           // learning rate
  unsigned nbBest;     // number of Best individuals used for update
  unsigned nbWorst;    // number of Worse individuals used for update
  double lrb, lrw;     // "local" learning rates (see operator())
};

#endif
