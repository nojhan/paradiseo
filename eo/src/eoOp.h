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

#ifndef _eoOp_H
#define _eoOp_H

#include <eoObject.h>
#include <eoPrintable.h>

/** @name Genetic operators

What is a genetic algorithm without genetic operators? There is a genetic operator hierarchy, with 
eoOp as father and eoMonOp (monary or unary operator) and eoBinOp and eoQuadraticOp (binary operators) 
as siblings). Nobody should subclass eoOp, you should subclass eoGeneralOp, eoBinOp, eoQuadraticOp or eoMonOp, 
those are the ones actually used here.\\

#eoOp#s are only printable objects, so if you want to build them from a file, it has to
be done in another class, namely factories. Each hierarchy of #eoOp#s should have its own
factory, which know how to build them from a description in a file. 
@author GeNeura Team
@version 0.1
@see eoOpFactory
*/


/** Abstract data types for EO operators.
 * Genetic operators act on chromosomes, changing them. The type to instantiate them should
 * be an eoObject, but in any case, they are type-specific; each kind of evolvable object
 * can have its own operators
 */
template<class EOType>
class eoOp: public eoObject, public eoPrintable {
public:
    
//@{
  enum OpType { unary = 0, binary = 1, quadratic = 2, general = 3};
///

  /// Ctor
  eoOp(OpType _type)
    :opType( _type ) {};

  /// Copy Ctor
  eoOp( const eoOp& _eop )
    :opType( _eop.opType ) {};

  /// Needed virtual destructor
  virtual ~eoOp(){};

  /// getType: number of operands it takes and individuals it produces
  OpType getType() const {return opType;};

	/** @name Methods from eoObject	*/
	//@{
  
  /**
   * Write object. It's called printOn since it prints the object _on_ a stream.
   * @param _os A ostream.
   */
  virtual void printOn(ostream& _os) const {
    _os << className().c_str(); 
    //	  _os << arity;
  };
  
  /** Inherited from eoObject 
      @see eoObject
  */
  virtual string className() const {return "eoOp";};
  //@}

private:
  /// OpType is the type of the operator: how many operands it takes and how many it produces
  OpType opType;

};

/** Binary genetic operator: subclasses eoOp, and defines
basically the operator() with two operands 
*/
template<class EOType>
class eoBinOp: public eoOp<EOType> {
public:

  /// Ctor
  eoBinOp()
    :eoOp<EOType>( binary ) {};

  /// Copy Ctor
  eoBinOp( const eoBinOp& _ebop )
    : eoOp<EOType>( _ebop ){};

  /// Dtor
  ~eoBinOp () {};

  /** applies operator, to the object. Modifies only the first operand.
   */
  virtual void operator()( EOType& _eo1, const EOType& _eo2 ) const = 0;

  /** @name Methods from eoObject
	readFrom and printOn are directly inherited from eoObject
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	virtual string className() const {return "eoBinOp";};
    //@}

};

/** Quadratic genetic operator: subclasses eoOp, and defines
basically the operator() with two operands 
*/
template<class EOType>
class eoQuadraticOp: public eoOp<EOType> {
public:

  /// Ctor
  eoQuadraticOp()
      :eoOp<EOType>( eoOp<EOType>::quadratic ) {};

  /// Copy Ctor
  eoQuadraticOp( const eoQuadraticOp& _ebop )
    : eoOp<EOType>( _ebop ){};

  /// Dtor
  ~eoQuadraticOp() {};

  /** applies operator, to the object. Modifies both operands.
   */
  virtual void operator()( EOType& _eo1, EOType& _eo2 ) const = 0;

  /** @name Methods from eoObject
	readFrom and printOn are directly inherited from eoObject
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	virtual string className() const {return "eoBinOp";};
    //@}

};

/** eoMonOp is the monary operator: genetic operator that takes
 only one EO
*/
template <class EOType>
class eoMonOp: public eoOp<EOType> {
public:

  /// Ctor
  eoMonOp( )
      : eoOp<EOType>( eoOp<EOType>::unary ) {};

  /// Copy Ctor
  eoMonOp( const eoMonOp& _emop )
    : eoOp<EOType>( _emop ){};

  /// Dtor
  ~eoMonOp() {};

  /** applies randomly operator, to the object. If arity is more than 1,
   * keeps a copy of the operand in a cache.
   */
  virtual void operator()( EOType& _eo1) const = 0;

  /** @name Methods from eoObject
	readFrom and printOn are directly inherited from eoObject
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	virtual string className() const {return "eoMonOp";};
    //@}  
};

// some forward declarations
template<class EOT>
class eoIndiSelector;

template<class EOT>
class eoInserter;

/**
 * eGeneralOp: General genetic operator; for objects used to transform sets
 of EOs. Nary ("orgy") operators should be derived from this class
*/
template<class EOT>
class eoGeneralOp: public eoOp<EOT>
{
public:

  /// Ctor that honors its superclass
    eoGeneralOp(): eoOp<EOT>( eoOp<EOT>::general ) {};

  /// Virtual dtor
  virtual ~eoGeneralOp () {};

  /** Method that really does the stuff. Applies the genetic operator
      to a individuals dispensed by an eoIndividualSelector, 
        and puts the results in the eoIndividualInserter.
        Any number of inputs can be requested and any number of outputs
        can be produced. 
  */
  virtual void operator()( eoIndiSelector<EOT>& _in, 
			   eoInserter<EOT>& _out) const = 0;
  
  virtual string className() const {return "eoGeneralOp";};
};


#endif

