// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-



//-----------------------------------------------------------------------------

// eoRank.h

// (c) GeNeura Team 1999

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



#ifndef _eoRank_H

#define _eoRank_H


#include <eoOpSelector.h>

#include <eoPopOps.h>


/**
 * Takes those on the selection std::list and creates a std::list of new individuals
 * Destroys the genetic pool. There's no requisite on EOT, other than the 
 * genetic operators can be instantiated with it, so it fully depstd::ends on 
 * the genetic operators used. If generic genetic operators are used, then 
 * EOT must be an EO 
 */

template<class EOT>
class eoRank: public eoSelect<EOT>, public eoObject, public eoPrintable
{
 public:
  
  /// Ctor
  eoRank( unsigned _newPopSize, eoOpSelector<EOT>& _opSelector)
	  :eoSelect<EOT>(), opSelector( _opSelector ), repNewPopSize( _newPopSize ) {};
    
  /// Dtor
  virtual ~eoRank() {};
  
  /** Takes the genetic pool, and returns next generation, destroying the
   * genetic pool container
   * Non-const because it might order the operator std::vector*/
  virtual void operator() (	const eoPop< EOT >& _ptVeo, 

							eoPop< EOT >& _siblings  ) const { 
    
    unsigned inLen = _ptVeo.size(); // size of subPop
    if ( !inLen ) 
      throw std::runtime_error( "zero population in eoRank");

    for ( unsigned i = 0; i < repNewPopSize; i ++ ) {
      // Create a copy of a random input EO with copy ctor. The members of the
		// population will be selected by rank, with a certain probability of
		// being selected several times if the new population is bigger than the
		// old

		EOT newEO =  _ptVeo[ i%inLen ];
    
		// Choose operator
		
        const eoOp<EOT >& thisOp = opSelector.Op();
		if ( thisOp.readArity() == unary ) {
			const eoMonOp<EOT>& mopPt = dynamic_cast< const eoMonOp<EOT>& > ( thisOp );
			mopPt( newEO );
		} else {
			const eoBinOp<EOT>& bopPt = dynamic_cast< const eoBinOp<EOT>& > ( thisOp );

			EOT mate =  _ptVeo[ rng.random(inLen) ];
			bopPt( newEO, mate );
		}      
		

		_siblings.push_back( newEO );
    }
  };

  /** This is a _new_ function, non defined in the parent class
   * It´s used to set the selection rate */
  void select( unsigned _select ) {
    repNewPopSize = _select;
  }


  

      /// Methods inherited from eoObject

    //@{



    /** Return the class id. 

      @return the class name as a std::string

      */

    virtual std::string className() const { return "eoRank"; };

    virtual void printOn( std::ostream& _s ) const
    {
  		_s << repNewPopSize;
    };





    //@}


private:
  eoOpSelector<EOT> & opSelector;
  unsigned repNewPopSize;
  
};

#endif
