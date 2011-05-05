// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_PBILupdate.h
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

#ifndef _make_PBILupdate_h
#define _make_PBILupdate_h

#include <ga/eoPBILOrg.h>
#include <utils/eoRNG.h>
#include <utils/eoParser.h>
#include <utils/eoState.h>


///////CONSTRUCTION of PBIL distribution updaters/////////////////
/**
 * Templatized version of parser-based construct of the update rule for PBIL
 * distributions (see eoPBILDistrib.h)
 *
 * It must then be instantiated, and compiled on its own for a given EOType
 * (see  test/t-eoPBIL.cpp)
 *
 * Last argument is template-disambiguating
*/


template <class EOT>
eoDistribUpdater<EOT> &  do_make_PBILupdate(eoParser & _parser, eoState& _state, EOT)
{
  // ast the moment, a single update rule is available
  // but at some point we'll have to choose among different rules

    std::string UType = _parser.createParam(std::string("PBIL"), "updateRule", "Type of update rule (only PBIL additive at the moment)", 'U', "Algorithm").value();

  unsigned nbBest = _parser.createParam(unsigned(1), "nbBest", "Number of good guys to update from", 'B', "Algorithm").value();
    double LRBest = _parser.createParam(0.1, "LRBest", "Learning Rate (from best guys)", 'L', "Algorithm").value();
  unsigned nbWorst = _parser.createParam(unsigned(0), "nbWorst", "Number of bad guys to update from", 'W', "Algorithm").value();
    double LRWorst = _parser.createParam(0.01, "LRWorst", "Learning Rate (from best guys)", 'L', "Algorithm").value();
    double tolerance = _parser.createParam(0.0, "tolerance", "Keeping away from 0 and 1", 't', "Algorithm").value();

    // a pointer to choose among several possible types
  eoDistribUpdater<EOT> * ptUpdate;

  if (UType == std::string("PBIL"))
    {
    if ( (nbWorst == 0) && (nbBest == 1) )
      ptUpdate = new eoPBILOrg<EOT>(LRBest, tolerance);
    else
      ptUpdate = new eoPBILAdditive<EOT>(LRBest, nbBest, tolerance, LRWorst, nbWorst);
    }
  else
    throw std::runtime_error("Only PBIL additive update rule available at the moment");

  // done: don't forget to store the allocated pointers
  _state.storeFunctor(ptUpdate);
  return *ptUpdate;
}

#endif
