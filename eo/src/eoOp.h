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

#include <vector>
#include <eoObject.h>
#include <eoPrintable.h>

/** @name Genetic operators

What is a genetic algorithm without genetic operators? There is a genetic operator hierarchy, with 
eoOp as father and eoMonOp (monary or unary operator) and eoBinOp (binary operator) as siblings. Nobody
should subclass eoOp, you should subclass eoBinOp or eoMonOp, those are the ones actually used here.\\
#eoOp#s are only printable objects, so if you want to build them from a file, it has to
be done in another class, namely factories. Each hierarchy of #eoOp#s should have its own
factory, which know how to build them from a description in a file. 
@author GeNeura Team
@version 0.1
@see eoOpFactory
*/
//@{

///
enum Arity { unary = 0, binary = 1, Nary = 2};

/** Abstract data types for EO operators.
 * Genetic operators act on chromosomes, changing them. The type to instantiate them should
 * be an eoObject, but in any case, they are type-specific; each kind of evolvable object
 * can have its own operators
 */
template<class EOType>
class eoOp: public eoObject, public eoPrintable {
public:

  /// Ctor
  eoOp( Arity _arity = unary )
    :arity( _arity ) {};

  /// Copy Ctor
  eoOp( const eoOp& _eop )
    :arity( _eop.arity ) {};

  /// Needed virtual destructor
  virtual ~eoOp(){};

  /// Arity: number of operands
  Arity readArity() const {return arity;};

	/** @name Methods from eoObject	*/
	//@{
  
  /**
   * Write object. It's called printOn since it prints the object _on_ a stream.
   * @param _os A ostream.
   */
  virtual void printOn(ostream& _os) const {
    _os << className(); 
    //	  _os << arity;
  };
  
  /** Inherited from eoObject 
      @see eoObject
  */
  virtual string className() const {return "eoOp";};
  //@}

private:
  /// arity is the number of operands it takes
  Arity arity;

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
    :eoOp<EOType>( unary ) {};

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

#include <eoPop.h>
/** eoNaryOp is the N-ary operator: genetic operator that takes
 several EOs. It could be called an {\em orgy} operator. It's a general operator
 that takes any number of inputs and spits out any number of outputs
*/
template <class EOType>
class eoNaryOp: public eoOp<EOType> {
public:

  /// Ctor
  eoNaryOp( )
    :eoOp<EOType>( Nary ) {};
  
  /// Copy Ctor
  eoNaryOp( const eoNaryOp& _emop )
    : eoOp<EOType>( _emop ){};
  
  /// Dtor
  ~eoNaryOp() {};
  
  /** applies randomly operator, to the object.
   */
  virtual void operator()( const eoPop<EOType> & _in, eoPop<EOType> _out ) const = 0;
  
  /** @name Methods from eoObject
      readFrom and printOn are directly inherited from eoObject.
  */
  //@{
  /** Inherited from eoObject 
      @see eoObject
  */
  string className() const {return "eoNaryOp";};
  //@}

};
//@}

#endif
