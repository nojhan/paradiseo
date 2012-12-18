/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2012 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/


#ifndef _eoRankMuSelect_h
#define _eoRankMuSelect_h

#include "eoDetSelect.h"

/** Selects the "Mu" bests individuals.
 *
 * Note: sorts the population before trucating it.
 *
 * @ingroup Selectors
*/
template<typename EOT>
class eoRankMuSelect : public eoDetSelect<EOT>
{
public :
    // false, because mu is not a rate
    eoRankMuSelect( unsigned int mu ) : eoDetSelect<EOT>( mu, false ) {}

    void operator()(const eoPop<EOT>& source, eoPop<EOT>& dest)
    {
        eoPop<EOT> tmp( source );
        tmp.sort();
        eoDetSelect<EOT>::operator()( tmp, dest );
    }
};

#endif // !_eoRankMuselect_h
