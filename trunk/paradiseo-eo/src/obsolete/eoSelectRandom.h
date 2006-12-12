// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSelectRandom.h
// (c) GeNeura Team, 1998 - EEAAX 1999, Maarten Keijzer 2000
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
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoSelectRandom_h
#define eoSelectRandom_h

//-----------------------------------------------------------------------------

#include <utils/eoRNG.h>
#include <eoSelectOne.h>

//-----------------------------------------------------------------------------
/** eoSelectRandom: a selection method that selects ONE individual randomly
 -MS- 22/10/99 */
//-----------------------------------------------------------------------------

template <class EOT> class eoSelectRandom: public eoSelectOne<EOT> 
{
 public:
  
  /// not a big deal!!!
  virtual const EOT& operator()(const eoPop<EOT>& pop) 
  {
    return pop[rng.random(pop.size())] ;
  }
};

#endif eoSelectRandom_h

