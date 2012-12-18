// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoPBILOrg.h
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

#ifndef _eoPBILOrg_H
#define _eoPBILOrg_H

#include <eoDistribUpdater.h>
#include <ga/eoPBILDistrib.h>

/**
 * Distribution Class for PBIL algorithm
 *      (Population-Based Incremental Learning, Baluja and Caruana 95)
 *
 * This class implements the update rule from the original paper:
 *
 *  p(i)(t+1) = (1-LR)*p(i)(t) + LR*best(i)
*/

template <class EOT>
class eoPBILOrg :  public eoDistribUpdater<EOT>
{
public:
  /** Ctor with size of genomes, and update parameters */
  eoPBILOrg(double _LR, double _tolerance=0.0 ) :
    LR(_LR), maxBound(1.0-_tolerance), minBound(_tolerance)
  {}


  /** Update the distribution from the current population */
  virtual void operator()(eoDistribution<EOT> & _distrib, eoPop<EOT>& _pop)
  {
    const EOT & best = _pop.best_element();
    eoPBILDistrib<EOT>& distrib = dynamic_cast<eoPBILDistrib<EOT>&>(_distrib);

    std::vector<double> & p = distrib.value();

    for (unsigned g=0; g<distrib.size(); g++)
      {
        //	double & r = value()[g];
        p[g] *= (1-LR);
        if ( best[g] )
          p[g] +=  LR;
        // else nothing

        // stay away from 0 and 1
        p[g] = std::min(maxBound, p[g]);
        p[g] = std::max(minBound, p[g]);
      }
  }

private:
  double LR;    // learning rate for best guys
  double maxBound, minBound;    // proba stay away from 0 and 1 by at least tolerance
};

#endif
