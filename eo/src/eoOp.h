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
#include <eoFunctor.h>
#include <utils/eoRNG.h>
/**
\defgroup operators
Genetic Operators are used for various purposes
*/

/** @name Genetic operators

What is a genetic algorithm without genetic operators? There is a genetic operator hierarchy, with eoOp as father and eoMonOp (monary or unary operator) and eoBinOp and eoQuadraticOp (binary operators) as siblings). Nobody should subclass eoOp, you should subclass eoGeneralOp, eoBinOp, eoQuadraticOp or eoMonOp, those are the ones actually used here. \\#eoOp#s are only printable objects, so if you want to build them from a file, it has to be done in another class, namely factories. Each hierarchy of #eoOp#s should have its own factory, which know how to build them from a description in a file.

@author GeNeura Team
@version 0.1
@see eoOpFactory
*/


/** Abstract data types for EO operators.
 * Genetic operators act on chromosomes, changing them. The type to 
 * instantiate them should be an eoObject, but in any case, they are 
 * type-specific; each kind of evolvable object can have its own operators
 */
template<class EOType>

class eoOp
{
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

private:

  /// OpType is the type of the operator: how many operands it takes and how many it produces
  OpType opType;
};

/** eoMonOp is the monary operator: genetic operator that takes only one EO */
template <class EOType>
class eoMonOp: public eoOp<EOType>, public eoUF<EOType&, void>
{
public:
  /// Ctor
  eoMonOp()
    : eoOp<EOType>( eoOp<EOType>::unary ) {};
  virtual string className() const {return "eoMonOp";};
};


/** Binary genetic operator: subclasses eoOp, and defines basically the 
 *  operator() with two operands, only the first one can be modified
 */
template<class EOType>
class eoBinOp: public eoOp<EOType>, public eoBF<EOType&, const EOType&, void>
{
public:
  /// Ctor
  eoBinOp()
      :eoOp<EOType>( eoOp<EOType>::binary ) {};
  virtual string className() const {return "eoBinOp";};
};

// planning the change of name eoQuadraticOp --> eoQuadOp
#define eoQuadraticOp eoQuadOp

/** Quadratic genetic operator: subclasses eoOp, and defines basically the 
    operator() with two operands, both can be modified.
*/
template<class EOType>
class eoQuadraticOp: public eoOp<EOType>, public eoBF<EOType&, EOType&, void> {
public:
  /// Ctor
  eoQuadraticOp()
    :eoOp<EOType>( eoOp<EOType>::quadratic ) {};
  virtual string className() const {return "eoQuadOp";};
};

/** Turning an eoQuadOp into an eoBinOp: simply don't touch the second arg!
 */
template <class EOT>
class eoQuad2BinOp: public eoBinOp<EOT>
{
public:
  /** Ctor
   * @param _quadOp the eoQuadOp to be transformed
   */
  eoQuad2BinOp(eoQuadOp<EOT> & _quadOp) : quadOp(_quadOp) {}

  /** Operator() simply calls embedded quadOp operator() with dummy second arg
   */
  void operator()(EOT & _eo1, const EOT & _eo2)
  {
    EOT eoTmp = _eo2;		   // a copy that can be modified
    // if the embedded eoQuadOp is not symmetrical, 
    // the result might be biased - hence the flip ...
    if (eo::rng.flip(0.5))
      quadOp(_eo1, eoTmp);	   // both are modified - that's all
    else
      quadOp(eoTmp, _eo1);	   // both are modified - that's all
  }

private:
  eoQuadOp<EOT> & quadOp;
};

// some forward declarations

template<class EOT>

class eoIndiSelector;


template<class EOT>

class eoInserter;


/**
 * eGeneralOp: General genetic operator; for objects used to transform sets
 * of EOs. Nary ("orgy") operators should be derived from this class

  Derived from eoB(inary)F(unction)
      Applies the genetic operator
      to a individuals dispensed by an eoIndividualSelector, 
      and puts the results in the eoIndividualInserter.
      Any number of inputs can be requested and any number of outputs
      can be produced. 
 */

template<class EOT>

class eoGeneralOp: public eoOp<EOT>, public eoBF<eoIndiSelector<EOT>&, eoInserter<EOT>&, void>
{
public:
  /// Ctor that honors its superclass
  eoGeneralOp(): eoOp<EOT>( eoOp<EOT>::general ) {}  
};

#endif

