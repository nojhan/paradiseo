// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoUniformSelect.h
// (c) GeNeura Team, 1998 - EEAAX 1999
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
 */
//-----------------------------------------------------------------------------

#ifndef eoUniformSelect_h
#define eoUniformSelect_h
// WARNING: 2 classes in this one - eoUniformSelect and eoCopySelect

//-----------------------------------------------------------------------------

#include <functional>  // 
#include <numeric>     // accumulate
#include <eoPopOps.h>          // eoPop eoSelect MINFLOAT
#include <eoRNG.h>

//-----------------------------------------------------------------------------
/** eoUniformSelect: a selection method that selects ONE individual randomly
 -MS- 22/10/99 */
//-----------------------------------------------------------------------------

template <class EOT> class eoUniformSelect: public eoSelectOne<EOT>
{
 public:
  /// (Default) Constructor.
  eoUniformSelect():eoSelectOne<EOT>() {}
  
  /// not a big deal!!!
  virtual const EOT& operator()(const eoPop<EOT>& pop) {
    return pop[rng.random(pop.size())] ;
  }
  
  /// Methods inherited from eoObject
  //@{
  
  /** Return the class id. 
   *  @return the class name as a string
   */
  virtual string className() const { return "eoUniformSelect"; };

 private:
};

#endif eoUniformSelect_h

