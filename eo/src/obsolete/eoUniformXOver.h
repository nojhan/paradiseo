// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-



//-----------------------------------------------------------------------------

// eoUniformXOver.h

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



#ifndef _EOUNIFORMXOVER_h

#define _EOUNIFORMXOVER_h





// for swap

#if defined( __BORLANDC__ )

#include <algorith>

#else

#include <algorithm>

#endif



// EO includes

#include <eoOp.h>

#include <utils/eoRNG.h>



//-----------------------------------------------------------------------------

/** 

 * EOUniformCrossover: operator for binary chromosomes

 * implementation of uniform crossover for EO

 * swaps ranges of bits between the parents

 */

//-----------------------------------------------------------------------------



template<class EOT> 

class eoUniformXOver: public eoQuadraticOp< EOT >

{

 public:



  ///

  eoUniformXOver( float _rate = 0.5 ): 

    eoQuadraticOp< EOT > (  ), rate( _rate ) {

    if (rate < 0 || rate > 1)

      std::runtime_error("UxOver --> invalid rate");

  }

  

  

  ///

  void operator() ( EOT& chrom1, EOT& chrom2 ) const {

    unsigned end = min(chrom1.length(),chrom2.length()) - 1;

    // select bits to change

    // aply changes

    for (unsigned bit = 0; bit < end; bit++)

      if (rng.flip(rate))

	swap(chrom1[ bit], chrom2[ bit]);

  }



  /** @name Methods from eoObject

      readFrom and printOn are directly inherited from eoOp

  */

  //@{

  /** Inherited from eoObject 

      @see eoObject

  */

  std::string className() const {return "eoUniformXOver";};

  //@}



private:

  float rate; /// rate of uniform crossover

};



//-----------------------------------------------------------------------------





#endif

