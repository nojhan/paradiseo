// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoMerge.h
//   Base class for elitist-merging classes
// (c) GeNeura Team, 1998
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
 */
//-----------------------------------------------------------------------------

#ifndef eoMerge_h
#define eoMerge_h

//-----------------------------------------------------------------------------

#include <stdexcept>

// EO includes
#include <eoPop.h>     // eoPop
#include <eoFunctor.h>  // eoMerge

/**
 * eoMerge: Base class for elitist replacement algorithms. 
 * Merges the old population (first argument), with the new generation
 *
 * Its signature is exactly
 * that of the selection base eoSelect, but its purpose is to merge the 
 * two populations into one (the second argument).
 * Note that the algorithms assume that the second argument denotes the 
 * next generation.
*/

template<class Chrom> class eoMerge: public eoBF<const eoPop<Chrom>&, eoPop<Chrom>&, void>
{};

/**
Straightforward elitism class, specify the number of individuals to copy
into new geneneration
*/
template <class EOT> class eoElitism : public eoMerge<EOT>
{
    public :
        eoElitism(unsigned _howmany) : howmany(_howmany) {}

        void operator()(const eoPop<EOT>& _pop, eoPop<EOT>& offspring)
        {
            if (howmany == 0)
                return;

            if (howmany > _pop.size())
                throw std::logic_error("Elite larger than population");

            vector<const EOT*> result;
            _pop.nth_element(howmany, result);

            for (size_t i = 0; i < result.size(); ++i)
            {
                offspring.push_back(*result[i]);
            }
        }

    private :
        unsigned howmany;
};

/**
No elite
*/
template <class EOT> class eoNoElitism : public eoElitism<EOT>
{
    public :
        eoNoElitism() : eoElitism<EOT>(0) {}
};

/**
Very elitist class, copies entire population into next gen
*/
template <class EOT> class eoPlus : public eoMerge<EOT>
{
    public :
        void operator()(const eoPop<EOT>& _pop, eoPop<EOT>& offspring)
        {
            offspring.reserve(offspring.size() + _pop.size());

            for (size_t i = 0; i < _pop.size(); ++i)
            {
                offspring.push_back(_pop[i]);
            }
        }

    private :
        unsigned howmany;
};

//-----------------------------------------------------------------------------

#endif 
