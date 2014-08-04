// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; fill-column: 80; -*-

//-----------------------------------------------------------------------------
// eoCMABreed
// (c) Maarten Keijzer 2005
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

 */
//-----------------------------------------------------------------------------


#ifndef _EOCMABREED_H
#define _EOCMABREED_H

#include <eoBreed.h>
#include <eoVector.h>
#include <es/CMAState.h>

#include <algorithm>

/// @todo handle bounds
template <class FitT>
class eoCMABreed : public eoBreed< eoVector<FitT, double> > {

    eo::CMAState& state;
    unsigned lambda;

    typedef eoVector<FitT, double> EOT;

    public:
    eoCMABreed(eo::CMAState& state_, unsigned lambda_) : state(state_), lambda(lambda_) {}

    void operator()(const eoPop<EOT>& parents, eoPop<EOT>& offspring) {

        // two temporary arrays of pointers to store the sorted population
        std::vector<const EOT*> sorted(parents.size());

        // mu stores population as vector (instead of eoPop)
        std::vector<const std::vector<double>* > mu(parents.size());

        parents.sort(sorted);
        for (unsigned i = 0; i < sorted.size(); ++i) {
            mu[i] = static_cast< const std::vector<double>* >( sorted[i] );
        }

        // learn
        state.reestimate(mu, sorted[0]->fitness(), sorted.back()->fitness());

        if (!state.updateEigenSystem(10)) {
            std::cerr << "No good eigensystem found" << std::endl;
        }

        // generate
        offspring.resize(lambda);

        for (unsigned i = 0; i < lambda; ++i) {
            state.sample( static_cast< std::vector<double>& >( offspring[i] ));
            offspring[i].invalidate();
        }

    }
};


#endif
