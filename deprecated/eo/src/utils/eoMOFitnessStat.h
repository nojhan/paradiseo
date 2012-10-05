// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFitnessStat.h
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

#ifndef _eoFitnessStat_h
#define _eoFitnessStat_h

#include <utils/eoStat.h>

/**
    The fitnesses of a whole population, as a vector
*/
template <class EOT, class FitT = typename EOT::Fitness>
class eoFitnessStat : public eoSortedStat<EOT, std::vector<FitT> >
{
public :

    using eoSortedStat<EOT, std::vector<FitT> >::value;

    eoFitnessStat(std::string _description = "AllFitnesses") :
      eoSortedStat<EOT,std::vector<FitT> >(std::vector<FitT>(0), _description) {}

    virtual void operator()(const std::vector<const EOT*>& _popPters)
    {
      value().resize(_popPters.size());
      for (unsigned i=0; i<_popPters.size(); i++)
        value()[i] = _popPters[i]->fitness();
    }
};


/** For multi-objective fitness, we need to translate a stat<vector<double> >
    into a vector<stat>, so each objective gets a seperate stat
*/
#ifdef _MSC_VER
// The follownig is needed to avoid some bug in Visual Studio 6.0
typedef double PartFitDefault;
template <class EOT, class PartFitT = PartFitDefault>
class eoMOFitnessStat : public eoSortedStat<EOT, std::vector<PartFitT> >
#else
template <class EOT, class PartFitT = double>
class eoMOFitnessStat : public eoSortedStat<EOT, std::vector<PartFitT> >
#endif

{
public:

    using eoSortedStat<EOT, std::vector<PartFitT> >::value;

  /** Ctor: say what component you want
   */
  eoMOFitnessStat(unsigned _objective, std::string _description = "MO-Fitness") :
    eoSortedStat<EOT,  std::vector<PartFitT> >(std::vector<PartFitT>(0), _description),
    objective(_objective) {}

    virtual void operator()(const std::vector<const EOT*>& _popPters)
    {
      value().resize(_popPters.size());

      for (unsigned i=0; i<_popPters.size(); i++)
      {
        value()[i] = _popPters[i]->fitness()[objective];
      }
    }
private:
  unsigned int objective;                  // The objective we're storing

};
#endif
