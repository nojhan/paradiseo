// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoScalarFitnessStat.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2000, 2001
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

#ifndef _eoScalarFitnessStat_h
#define _eoScalarFitnessStat_h

#include <utils/eoRealVectorBounds.h>
#include <utils/eoStat.h>

/**
    The fitnesses of a whole population, as a std::vector

    @ingroup Stats
*/
template <class EOT, class FitT = typename EOT::Fitness>
class eoScalarFitnessStat : public eoSortedStat<EOT, std::vector<double> >
{
public:

    using eoSortedStat<EOT, std::vector<double> >::value;

    eoScalarFitnessStat(std::string _description = "FitnessES",
                        eoRealVectorBounds & _bounds = eoDummyVectorNoBounds)
        : eoSortedStat<EOT,  std::vector<double> >(std::vector<double>(0), _description),
          range(*_bounds[0])
        {}

  virtual void operator()(const std::vector<const EOT*>& _popPters)
    {
      value().resize(_popPters.size());
      for (unsigned i=0; i<_popPters.size(); i++)
        {
          value()[i] = _popPters[i]->fitness();
          range.truncate(value()[i]);
        }
    }

private :
  eoRealBounds & range;
};

#endif
