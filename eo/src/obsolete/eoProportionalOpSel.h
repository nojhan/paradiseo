// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-



//-----------------------------------------------------------------------------

// eoProportionalOpSel.h

// (c) GeNeura Team 1998
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

#ifndef EOPROPORTIONALOPSEL_H
#define EOPROPORTIONALOPSEL_H

//-----------------------------------------------------------------------------

#include <stdexcept>  // runtime_error
#include <functional>   // greater 
#include <map>

// Includes from EO
#include <utils/eoRNG.h>
#include <eoOpSelector.h>
#include <eoOp.h>

//-----------------------------------------------------------------------------
/** This class selects operators according to probability. All operator percentages
should add up to one; if not, an exception will be raised.\\
Operators are represented as pairs (proportion,operator)
*/
template<class EOT>
class eoProportionalOpSel: public eoOpSelector<EOT>, 
	public  multimap<float,eoOp<EOT>*,greater<float> >
{
public:

  typedef multimap<float, eoOp<EOT>*,greater<float> > MMF;

  /// default ctor
  eoProportionalOpSel()
    : eoOpSelector<EOT>(), MMF(), opID(1) {};

  /// virtual dtor
  virtual ~eoProportionalOpSel() {};

  /** Gets a non-const reference to an operator, so that it can be changed, 
    modified or whatever 
    @param _id  a previously assigned ID
    @throw runtime_error if the ID does not exist*/
 
  virtual eoOp<EOT>& getOp( ID _id ) {
     MMF::iterator i=begin();
     ID j = 1;
     while ( (i++!=end()) &&  (j++ != _id) );
     if ( i == end() ) 
       throw runtime_error( "No such id in eoProportionalOpSel::op\n" );
     return *(i->second);
	 //return i->second;
  }

  /** add an operator to the operator set
	@param _op a genetic operator, that will be applied in some way
	@param _arg an argument to the operator, usually operator rate
	@return an ID that will be used to identify the operator
	*/
  virtual ID addOp( eoOp<EOT>& _op, float _arg ) {
    insert( MMF::value_type( _arg,&             _op ) );
    return opID++;
  }	

  /** Remove an operator from the operator set
    @param _id a previously assigned ID
    @throw runtime_error if the ID does not exist
    */
  virtual void deleteOp( ID _id ) {
    unsigned j;
    MMF::iterator i;
    for ( i=begin(), j=1; i!=end(); i++,j++ ) {
      if( j == _id ) 
	erase( i );
      return;
    }
    if ( i == end() ) 
      throw runtime_error( "No such id in eoProportionalOpSel::op\n" );
  };

  /// Returns a genetic operator according to the established criteria
  virtual eoOp<EOT>* Op() {
    // Check that all add up to one
    float acc = 0;
    MMF::iterator i;
    unsigned j;
    for ( i=begin(), j=1; i!=end(); i++,j++ ) {
      acc +=i->first;
    }
    if ( acc != 1.0 )
      throw runtime_error( "Operator rates added up different from 1.0" );
	
    // If here, operators ordered by rate and no problem
    float aRnd = rng.uniform();
    i=begin();
    acc = 0;
    do {
      acc += i->first;
    } while ( (acc <= aRnd ) && (i++!=end() ) );
	if ( i == end() )
		throw runtime_error( "Operator not found in eoProportionalOpSelector" );
	return i->second;
	//return i->second;
  }

    /// Methods inherited from eoObject
    //@{

    /** Return the class id. 
      @return the class name as a string
      */
    virtual string className() const { return "eoProportionalOpSel"; };

    /** Print itself: inherited from eoObject implementation. Declared virtual so that 
      it can be reimplemented anywhere. Instance from base classes are processed in
      base classes, so you don´t have to worry about, for instance, fitness.
      @param _s the ostream in which things are written*/
    virtual void printOn( ostream& _s ) const{
		_s << className().c_str() << endl;
		for ( MMF::const_iterator i=begin(); i!=end(); i++ ) {
			_s << i->first << "\t" << *(i->second )<< endl;
		}
    }


    //@}

  private:
    ID opID;
  };

  //-----------------------------------------------------------------------------

#endif EO_H

