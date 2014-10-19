// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; fill-column: 80; -*-

//-----------------------------------------------------------------------------
// eoCMAInit
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

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             marc.schoenauer@polytechnique.fr
                       http://eeaax.cmap.polytchnique.fr/
 */
//-----------------------------------------------------------------------------


#ifndef _EOCMAINIT_H
#define _EOCMAINIT_H

#include <eoInit.h>
#include <eoVector.h>
#include <es/CMAState.h>

/// @todo handle bounds
template <class FitT>
class eoCMAInit : public eoInit< eoVector<FitT, double> > {

    const eo::CMAState& state;

    typedef eoVector<FitT, double> EOT;

    public:
    eoCMAInit(const eo::CMAState& state_) : state(state_) {}


    void operator()(EOT& v) {
        state.sample(static_cast<std::vector<double>& >(v));
        v.invalidate();
    }
};


#endif
