// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoOp.h
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

#ifndef _EOALGO_H
#define _EOALGO_H

#include <eoPop.h>                   // for population

/** This is a generic class for population-transforming algorithms. There
    is only one operator defined, which takes a population and does stuff to
    it. It needn't be a complete algorithm, can be also a step of an
    algorithm. This class just gives a common interface to linear
    population-transforming algorithms.
 @author GeNeura Team
 @version 0.0
*/
template< class EOT >
class eoAlgo {
public:

  /// Dtor
  virtual ~eoAlgo() {};
  
  /** Run the algorithm on a population. This operation is not constant, 
      because somebody would want to change stuff in the algorithm each
      time it's applied. 
   * @param _pop is the population that the algorithm is acting on
   */
  virtual void operator() ( eoPop<EOT>& _pop ) = 0;
  
};
	

#endif
