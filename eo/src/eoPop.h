// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoPop.h 
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

#ifndef _EOPOP_H
#define _EOPOP_H

#include <vector>
#include <strstream>

// EO includes
#include <eoRnd.h>
#include <eoPersistent.h>

/** Subpopulation: it is used to move parts of population
 from one algorithm to another and one population to another. It is safer
to declare it as a separate object. I have no idea if a population can be
some other thing that a vector, but if somebody thinks of it, this concrete
implementation will be moved to "generic" and an abstract Population 
interface will be provided.
It can be instantiated with anything, provided that it accepts a "size" and a 
random generator in the ctor. This happens to all the eo1d chromosomes declared 
so far. EOT must also have a copy ctor, since temporaries are created and copied
to the population.
@author Geneura Team
@version 0.0
*/

template<class EOT>
class eoPop: public vector<EOT>, public eoObject, public eoPersistent {

  /*
   Victor Rivas (vrivas@ujaen.es): 15-Dec-1999
   An EO can have NO genes.
   /// Type is the type of each gene in the chromosome
   #ifdef _MSC_VER
   typedef EOT::Type Type;
   #else
   typedef typename EOT::Type Type;
   #endif
  */
  
public:
  /** Protected ctor. This is intended to avoid creation of void populations, except 
      from sibling classes
  */
  eoPop():vector<EOT>() {};
  
  
  /** Ctor for fixed-size chromosomes, with variable content
      @param _popSize total population size
      @param _eoSize chromosome size. EOT should accept a fixed-size ctor
      @param _geneRdn random number generator for each of the genes
  */
  /*
    Victor Rivas (vrivas@ujaen.es): 15-Dec-1999
    This constructor must be substitued by one using factories.
  eoPop( unsigned _popSize, unsigned _eoSize, eoRnd<Type> & _geneRnd )
    :vector<EOT>() {
    for ( unsigned i = 0; i < _popSize; i ++ ){
      EOT tmpEOT( _eoSize, _geneRnd);
      push_back( tmpEOT );
    }
  };
  */
  
  /** Ctor for variable-size chromosomes, with variable content
      @param _popSize total population size
      @param _sizeRnd RNG for the chromosome size. This will be added 1, just in case.
      @param _geneRdn random number generator for each of the genes
  */
  /*
    Victor Rivas (vrivas@ujaen.es): 15-Dec-1999
    This constructor must be substitued by one using factories.
    eoPop( unsigned _popSize, eoRnd<unsigned> & _sizeRnd, eoRnd<Type> & _geneRnd )
    :vector<EOT>() {
    for ( unsigned i = 0; i < _popSize; i ++ ){
    unsigned size = 1 + _sizeRnd();
    EOT tmpEOT( size, _geneRnd);
    push_back( tmpEOT );
    }
    };
  */
  /** Ctor from an istream; reads the population from a stream,
      each element should be in different lines
      @param _is the stream
  */
  eoPop( istream& _is ):vector<EOT>() {
    readFrom( _is );
  }
  
  ///
  ~eoPop() {};
  
  /** @name Methods from eoObject	*/
  //@{
  /**
   * Read object. The EOT class must have a ctor from a stream;
   in this case, a strstream is used.
   * @param _is A istream.
   
   */
  virtual void readFrom(istream& _is) {
    while( _is ) {  // reads line by line, and creates an object per
      // line
      char line[MAXLINELENGTH];
      _is.getline( line, MAXLINELENGTH-1 );
      if (strlen( line ) ) {
	istrstream s( line );
	EOT thisEOT( s );
	push_back( thisEOT );      
      }
    }
  }
  
  /**
   * Write object. It's called printOn since it prints the object _on_ a stream.
   * @param _os A ostream. In this case, prints the population to
   standard output. The EOT class must hav standard output with cout,
   but since it should be an eoObject anyways, it's no big deal.
  */
  virtual void printOn(ostream& _os) const {
    copy( begin(), end(), ostream_iterator<EOT>( _os, "\n") );
  };
  
  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  virtual string className() const {return "eoPop";};
  //@}
  
protected:
  
};
#endif
