/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

-----------------------------------------------------------------------------
 eoRnd.h
 (c) GeNeura Team, 1998
 
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
/**
CVS Info: $Date: 2001-02-13 22:35:07 $ $Header: /home/nojhan/dev/eodev/eodev_cvs/eo/src/utils/Attic/eoRnd.h,v 1.1 2001-02-13 22:35:07 jmerelo Exp $ $Author: jmerelo $ $Log$
*/
#ifndef _EORND_H
#define _EORND_H

//-----------------------------------------------------------------------------

#include <eoObject.h>

//-----------------------------------------------------------------------------
// Class eoRnd
//-----------------------------------------------------------------------------

#include <stdlib.h>   // srand
#include <time.h>     // time
#include <stdexcept>  // runtime_error

//-----------------------------------------------------------------------------

#include <eoPersistent.h>

//-----------------------------------------------------------------------------
/** 
 * Base class for a family of random 'number' generators. These 'numbers'
 * can be anything, including full-fledged chromosomes.  
 */
template<class T>
class eoRnd
{
public:

  /** Main function: random generators act as functors, that return random numbers. 
	@return return a random number
	*/
  virtual T operator()() = 0;
  
};

#endif
