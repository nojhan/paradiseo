// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoAtomMutator.h
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
    CVS Info: $Date: 2001-03-21 12:10:13 $ $Header: /home/nojhan/dev/eodev/eodev_cvs/eo/src/Attic/eoAtomMutator.h,v 1.4 2001-03-21 12:10:13 jmerelo Exp $ $Author: jmerelo $ 
 */
//-----------------------------------------------------------------------------
#ifndef _EOATOMMUTATOR_H
#define _EOATOMMUTATOR_H

#include <eoFunctor.h>

/** Abstract base class for functors that modify a single element in an EO 
    that is composed of several atomic components. An atom would, for instance, flip
    a bit, or change a real number, or things like that. The header is completely
    empty and thus just provides a name rather than functionality.
*/

template <class T>
class eoAtomMutator: public eoUF<T&, bool> {};


#endif

