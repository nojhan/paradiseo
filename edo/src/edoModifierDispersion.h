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

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
    Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _edoModifierDispersion_h
#define _edoModifierDispersion_h

#include <eoPop.h>
#include <eoFunctor.h>

#include "edoModifier.h"

/** An semantic pseudo-interface for modifiers that updates dispersion parameters (like variance).
 *
 * @ingroup Modifiers
 */
template < typename D >
class edoModifierDispersion : public edoModifier< D >, public eoBF< D&, eoPop< typename D::EOType >&, void >
{
public:
    // virtual void operator() ( D&, eoPop< D::EOType >& )=0 (provided by eoBF< A1, A2, R >)
};

#endif // !_edoModifierDispersion_h
