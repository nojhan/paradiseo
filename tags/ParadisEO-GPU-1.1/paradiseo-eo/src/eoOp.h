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
    CVS Info: $Date: 2004-08-10 17:19:46 $ $Header: /home/nojhan/dev/eodev/eodev_cvs/eo/src/eoOp.h,v 1.29 2004-08-10 17:19:46 jmerelo Exp $ $Author: jmerelo $
 */
//-----------------------------------------------------------------------------

#ifndef _eoOp_H
#define _eoOp_H

#include <eoObject.h>
#include <eoPrintable.h>
#include <eoFunctor.h>
#include <utils/eoRNG.h>

/**
@defgroup Operators Evolutionary Operators

in EO, an operator is any functors that modifies objects and inherits from an eoOp.

Typically, a mutation is an operator that modifies an individual, and an algorithm is an operator that modifies a population.

In EO, there is a genetic operator hierarchy, with eoOp as father and
eoMonOp (monary or unary operator), eoBinOp and eoQuadOp (binary operators)
and eoGenOp (any number of inputs and outputs, see eoGenOp.h)
as subclasses.
Nobody should subclass eoOp, you should subclass eoGenOp, eoBinOp, eoQuadOp
or eoMonOp, those are the ones actually used here.

#eoOp#s are only printable objects, so if you want to build them
from a file, it has to be done in another class, namely factories.
Each hierarchy of #eoOp#s should have its own factory, which know
how to build them from a description in a file.

@author GeNeura Team, Marten Keijzer and Marc Schoenauer
@version 0.9
@see eoGenOp.h eoOpFactory
*/
//@{

/**
@defgroup Variators Variation operators
Variators are operators that modify individuals.

@defgroup Selectors Selection operators

Selectors are operators that select a subset of a population.

Example:
@include t-eoSelect.cpp


@defgroup Replacors Replacement operators

Replacors are operators that replace a subset of a population by another set of individuals.

Here is an example with several replacement operators:
@include t-eoReplacement.cpp
*/

/** Abstract data types for EO operators.
  Genetic operators act on chromosomes, changing them.
  The type to use them on is problem specific. If your genotype
  is a std::vector<bool>, there are operators that work specifically
  on std::vector<bool>, but you might also find that generic operators
  working on std::vector<T> are what you need.

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

/**
eoMonOp is the monary operator: genetic operator that takes only one EO.
When defining your own, make sure that you return a boolean value
indicating that you have changed the content.
*/
template <class EOType>
class eoMonOp: public eoOp<EOType>, public eoUF<EOType&, bool>
{
public:
  /// Ctor
  eoMonOp()
    : eoOp<EOType>( eoOp<EOType>::unary ) {};
  virtual std::string className() const {return "eoMonOp";};
};


/** Binary genetic operator: subclasses eoOp, and defines basically the
 *  operator() with two operands, only the first one can be modified
When defining your own, make sure that you return a boolean value
indicating that you have changed the content.
 */
template<class EOType>
class eoBinOp: public eoOp<EOType>, public eoBF<EOType&, const EOType&, bool>
{
public:
  /// Ctor
  eoBinOp()
      :eoOp<EOType>( eoOp<EOType>::binary ) {};
  virtual std::string className() const {return "eoBinOp";};
};

/** Quad genetic operator: subclasses eoOp, and defines basically the
    operator() with two operands, both can be modified.
When defining your own, make sure that you return a boolean value
indicating that you have changed the content.
*/
template<class EOType>
class eoQuadOp: public eoOp<EOType>, public eoBF<EOType&, EOType&, bool> {
public:
  /// Ctor
  eoQuadOp()
    :eoOp<EOType>( eoOp<EOType>::quadratic ) {};
  virtual std::string className() const {return "eoQuadOp";};
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
  bool operator()(EOT & _eo1, const EOT & _eo2)
  {
    EOT eoTmp = _eo2;              // a copy that can be modified
    // if the embedded eoQuadOp is not symmetrical,
    // the result might be biased - hence the flip ...
    if (eo::rng.flip(0.5))
      return quadOp(_eo1, eoTmp);          // both are modified - that's all
    else
      return quadOp(eoTmp, _eo1);          // both are modified - that's all
  }

private:
  eoQuadOp<EOT> & quadOp;
};

#endif

//@}
