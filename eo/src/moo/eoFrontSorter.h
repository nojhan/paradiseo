/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoNDSorting.h
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

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoFrontSorter_h
#define eoFrontSorter_h

#include <EO.h>
#include <algorithm>
#include <eoPop.h>
#include <cassert>

namespace detail {
// small helper structs to store the multi-objective information. To be used in the implementation 
        
    struct  FitnessInfo {
          std::vector<double> fitness;  // preprocessed fitness -> all maximizing
          unsigned index;               // index into population
    
          FitnessInfo() {}
          FitnessInfo(const std::vector<double>& fitness_, unsigned index_) : fitness(fitness_), index(index_) {}
    };
    
    struct CompareOn {
        unsigned dim;
        double tol;

        CompareOn(unsigned d, double t) : dim(d), tol(t) {}

        bool operator()(const FitnessInfo& a, const FitnessInfo& b) {
            return  a.fitness[dim] > b.fitness[dim] && fabs(a.fitness[dim] - b.fitness[dim]) > tol;
        }

    };
    
    extern void front_sorter_impl(std::vector<FitnessInfo>& fitness, std::vector< std::vector<FitnessInfo> >& fronts, double tol);

} // namespace detail

/**
 * Reassembles population into a set of fronts;
*/
template <class EOT>
class eoFrontSorter : public eoUF< const eoPop<EOT>&, const std::vector< std::vector<detail::FitnessInfo> >& >
{

    std::vector<detail::FitnessInfo > fitness;
    std::vector< std::vector<detail::FitnessInfo> > fronts;

  public :

    typedef typename EOT::Fitness::fitness_traits Traits;
    
    typedef std::vector< std::vector<detail::FitnessInfo> > front_t;

    const std::vector<std::vector<detail::FitnessInfo> >& operator()(const eoPop<EOT>& _pop)
    {
        fitness.resize(_pop.size());
        for (unsigned i = 0; i < _pop.size(); ++i) {
            fitness[i] = detail::FitnessInfo(_pop[i].fitness().normalized(), i);
        }

        detail::front_sorter_impl(fitness, fronts, Traits::tol());
        
        return fronts;
    }
    
    const std::vector<std::vector<detail::FitnessInfo> >& operator()(const std::vector<EOT*>& _pop)
    {
        fitness.resize(_pop.size());
        for (unsigned i = 0; i < _pop.size(); ++i) {
            fitness[i] = detail::FitnessInfo(_pop[i]->fitness().normalized(), i);
        }

        detail::front_sorter_impl(fitness, fronts, Traits::tol());
        
        return fronts;
    }
};


#endif
