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
/**
\defgroup PropCombined operators
Combination of same-type Genetic Operators - Proportional choice 
*/

/** @name PropCombined Genetic operators

This files contains the classes eoPropCombinedXXXOp (XXX in {Mon, Bin, Quad}) 
that allow to use more than a single operator of a specific class 
into an algorithm, chosing by a roulette wheel based on user-defined rates

@author Marc Schoenauer
@version 0.1
*/



/** eoMonOp is the monary operator: genetic operator that takes only one EO */

template <class EOT>
class eoPropCombinedMonOp: public eoMonOp<EOT>
{
public:
  /// Ctor
  eoPropCombinedMonOp(eoMonOp<EOT> & _first, double _rate)
  { 
    ops.push_back(&_first); 
    rates.push_back(_rate);
  }

virtual string className() const { return "eoPropCombinedMonOp"; }

void add(eoMonOp<EOT> & _op, double _rate, bool _verbose=false)
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
	cout << "In " << className() << "\n" ;
	for (i=0; i<ops.size(); i++)
	  cout << ops[i]->className() << " with rate " << 100*rates[i]/total << " %\n";
      }
  }

  void operator()(EOT & _indi)
  {
    unsigned what = rng.roulette_wheel(rates); // choose one op
    (*ops[what])(_indi);		   // apply it
  }
private:
std::vector<eoMonOp<EOT>*> ops;
std::vector<double> rates;
};


/** COmbined Binary genetic operator: 
 *  operator() has two operands, only the first one can be modified
 */
template <class EOT>
class eoPropCombinedBinOp: public eoBinOp<EOT>
{
public:
  /// Ctor
  eoPropCombinedBinOp(eoBinOp<EOT> & _first, double _rate)
  { 
    ops.push_back(&_first); 
    rates.push_back(_rate);
  }

virtual string className() const { return "eoPropCombinedBinOp"; }

void add(eoBinOp<EOT> & _op, double _rate, bool _verbose=false)
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
	cout << "In " << className() << "\n" ;
	for (i=0; i<ops.size(); i++)
	  cout << ops[i]->className() << " with rate " << 100*rates[i]/total << " %\n";
      }
  }

  void operator()(EOT & _indi1, const EOT & _indi2)
  {
    unsigned what = rng.roulette_wheel(rates); // choose one op index
    (*ops[what])(_indi1, _indi2);		   // apply it
  }
private:
std::vector<eoBinOp<EOT>*> ops;
std::vector<double> rates;
};


/** Quadratic genetic operator: subclasses eoOp, and defines basically the 
    operator() with two operands, both can be modified.
*/
/** COmbined Binary genetic operator: 
 *  operator() has two operands, only the first one can be modified
 */
template <class EOT>
class eoPropCombinedQuadOp: public eoQuadraticOp<EOT>
{
public:
  /// Ctor
  eoPropCombinedQuadOp(eoQuadraticOp<EOT> & _first, double _rate)
  { 
    ops.push_back(&_first); 
    rates.push_back(_rate);
  }

virtual string className() const { return "eoPropCombinedQuadOp"; }

void add(eoQuadraticOp<EOT> & _op, double _rate, bool _verbose=false)
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
	cout << "In " << className() << "\n" ;
	for (i=0; i<ops.size(); i++)
	  cout << ops[i]->className() << " with rate " << 100*rates[i]/total << " %\n";
      }
  }

  void operator()(EOT & _indi1, EOT & _indi2)
  {
    unsigned what = rng.roulette_wheel(rates); // choose one op index
    (*ops[what])(_indi1, _indi2);		   // apply it
  }
private:
std::vector<eoQuadraticOp<EOT>*> ops;
std::vector<double> rates;
};


// for General Ops, it's another story - see eoGOpSelector
#endif 

