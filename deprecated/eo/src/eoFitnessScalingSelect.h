// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFitnessScalingSelect.h
// (c) GeNeura Team, 1998, Maarten Keijzer 2000, Marc Schoenauer 2001
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
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoFitnessScalingSelect_h
#define eoFitnessScalingSelect_h

//-----------------------------------------------------------------------------

#include <eoSelectFromWorth.h>
#include <eoLinearFitScaling.h>

/** eoFitnessScalingSelect: select an individual proportional to the
 *  linearly scaled fitness that is computed by the private
 *  eoLinearFitScaling object
 *
 *  @ingroup Selectors
*/
template <class EOT>
class eoFitnessScalingSelect:  public eoRouletteWorthSelect<EOT, double>
{
public:
  /** Ctor:
   *  @param _p the selective pressure, should be in [1,2] (2 is the default)
   */
  eoFitnessScalingSelect(double _p = 2.0):
    eoRouletteWorthSelect<EOT, double>(scaling), scaling(_p) {}

private :
  eoLinearFitScaling<EOT> scaling;         // derived from eoPerf2Worth
};

#endif
