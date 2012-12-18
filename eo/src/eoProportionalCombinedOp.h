// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-
//-----------------------------------------------------------------------------
// eoCombinedOp.h
// (c) GeNeura Team, 1998, Marc Schoenauer, 2000
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

#ifndef _eoCombinedOp_H
#define _eoCombinedOp_H

#include <eoObject.h>
#include <eoPrintable.h>
#include <eoFunctor.h>
#include <eoOp.h>
#include <utils/eoRNG.h>
#include <utils/eoLogger.h>

/**
\defgroup Utilities Utilities

Classes that may help you write elegant code.
*/

/**
@defgroup Combination Operators combination

Classes that permits to combine several operators in a single one.

@ingroup Utilities
@{
*/

/** @name PropCombined Genetic operators

Combination of same-type Genetic Operators.

This files contains the classes eoPropCombinedXXXOp (XXX in {Mon, Bin, Quad})
that allow to use more than a single operator of a specific class
into an algorithm, chosing by a roulette wheel based on user-defined rates

@author Marc Schoenauer
@version 0.1
*/


//////////////////////////////////////////////////////
//// combined MonOp
//////////////////////////////////////////////////////

/** eoMonOp is the monary operator: genetic operator that takes only one EO

 * now accepts generic operators
*/

template <class EOT>
class eoPropCombinedMonOp: public eoMonOp<EOT>
{
public:
  /// Ctor from a "true" operator
  eoPropCombinedMonOp(eoMonOp<EOT> & _first, const double _rate)
  {
    ops.push_back(&_first);
    rates.push_back(_rate);
  }

  virtual std::string className() const { return "eoPropCombinedMonOp"; }

  virtual void add(eoMonOp<EOT> & _op, const double _rate, bool _verbose=false)
  {
    ops.push_back(&_op);
    rates.push_back(_rate);
    // compute the relative rates in percent - to warn the user!
    if (_verbose)
      printOn( eo::log << eo::logging );
  }

  // outputs the operators and percentages
  virtual void printOn(std::ostream & _os)
  {
    double total = 0;
    unsigned i;
    for (i=0; i<ops.size(); i++)
      total += rates[i];
    _os << "In " << className() << "\n" ;
    for (i=0; i<ops.size(); i++)
      _os << ops[i]->className() << " with rate " << 100*rates[i]/total << " %\n";
  }

  virtual bool operator()(EOT & _indi)
  {
    unsigned what = rng.roulette_wheel(rates); // choose one op
    return (*ops[what])(_indi);            // apply it
  }
protected:
std::vector<eoMonOp<EOT>*> ops;
std::vector<double> rates;
};

//////////////////////////////////////////////////////
//// combined BinOp
//////////////////////////////////////////////////////

/** COmbined Binary genetic operator:
 *  operator() has two operands, only the first one can be modified
 */
template <class EOT>
class eoPropCombinedBinOp: public eoBinOp<EOT>
{
public:
  /// Ctor
  eoPropCombinedBinOp(eoBinOp<EOT> & _first, const double _rate)
  {
    ops.push_back(&_first);
    rates.push_back(_rate);
  }

virtual std::string className() const { return "eoPropCombinedBinOp"; }

virtual void add(eoBinOp<EOT> & _op, const double _rate, bool _verbose=false)
  {
    ops.push_back(&_op);
    rates.push_back(_rate);
    // compute the relative rates in percent - to warn the user!
    if (_verbose)
      {
        double total = 0;
        unsigned i;
        for (i=0; i<ops.size(); i++)
          total += rates[i];
    eo::log << eo::logging << "In " << className() << std::endl ;
        for (i=0; i<ops.size(); i++)
          eo::log << eo::logging << ops[i]->className() << " with rate " << 100*rates[i]/total << " %" << std::endl;
      }
  }

  virtual void operator()(EOT & _indi1, const EOT & _indi2)
  {
    unsigned what = rng.roulette_wheel(rates); // choose one op index
    return (*ops[what])(_indi1, _indi2);                   // apply it
  }
private:
std::vector<eoBinOp<EOT>*> ops;
std::vector<double> rates;
};


//////////////////////////////////////////////////////
//// combined QuadOp
//////////////////////////////////////////////////////

/** Quad genetic operator: subclasses eoOp, and defines basically the
    operator() with two operands, both can be modified.
*/
/** Combined quad genetic operator:
 *  operator() has two operands, both can be modified

 * generic operators are now allowed: there are imbedded into
 * the corresponding "true" operator
 */
template <class EOT>
class eoPropCombinedQuadOp: public eoQuadOp<EOT>
{
public:
  /// Ctor from a true operator
  eoPropCombinedQuadOp(eoQuadOp<EOT> & _first, const double _rate)
  {
    ops.push_back(&_first);
    rates.push_back(_rate);
  }

  virtual std::string className() const { return "eoPropCombinedQuadOp"; }

  /* FIXME remove in next release
  virtual void add(eoQuadOp<EOT> & _op, const double _rate, bool _verbose)
  {
#ifndef DEPRECATED_MESSAGES
#pragma message "The use of the verbose parameter in eoPropCombinedQuadOp::add is deprecated and will be removed in the next release."
    eo::log << eo::warnings << "WARNING: the use of the verbose parameter in eoPropCombinedQuadOp::add is deprecated and will be removed in the next release." << std::endl;
#endif // !DEPRECATED_MESSAGES

    add(_op,_rate);
  }
  */

  // addition of a true operator
  virtual void add(eoQuadOp<EOT> & _op, const double _rate)
  {
    ops.push_back(&_op);
    rates.push_back(_rate);
    // compute the relative rates in percent - to warn the user!
    printOn( eo::log << eo::logging );
  }

  // outputs the operators and percentages
  virtual void printOn(std::ostream & _os)
  {
    double total = 0;
    unsigned i;
    for (i=0; i<ops.size(); i++)
      total += rates[i];
    _os << "In " << className() << "\n" ;
    for (i=0; i<ops.size(); i++)
      _os << ops[i]->className() << " with rate " << 100*rates[i]/total << " %\n";
  }

  virtual bool operator()(EOT & _indi1, EOT & _indi2)
  {
    unsigned what = rng.roulette_wheel(rates); // choose one op index
    return (*ops[what])(_indi1, _indi2);                   // apply it
  }
private:
std::vector<eoQuadOp<EOT>*> ops;
std::vector<double> rates;
};


// for General Ops, it's another story - see eoOpContainer
#endif

/** @} */
