// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFlOrBinOp.h
// (c) Marc Schoenauer - Maarten Keijzer 2000-2003
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

    Contact: Marc.Schoenauer@inria.fr
             mkeijzer@cs.vu.nl
 */
//-----------------------------------------------------------------------------

#ifndef _eoFlOrBinOp_h
#define _eoFlOrBinOp_h

#include <eoFunctor.h>
#include <eoOp.h>

/** @addtogroup Variators
 * @{
 */

/** Generic eoBinOps on fixed length genotypes.
 *  Contains exchange crossovers (1pt and uniform)
 *      and crossovers that applies an Atom crossover
 *          to all components with given rate, or
 *          to a fixed prescribed nb of components
 *
 * Example: the standard bitstring 1-point and uniform crossovers
 *          could be implemented as resp. eoFlOr1ptBinOp and eoFlOrUniformBinOp
*/

//////////////////////////////////////////////////////////////////////
//                eoFlOrAllAtomBinOp
//////////////////////////////////////////////////////////////////////
/** Bin Crossover using an Atom Crossover
 *  that is applied to a ALL components with given rate
 */
template <class EOT>
class eoFlOrAllAtomBinOp : public eoBinOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;

  /** default ctor: requires an Atom BinOp */
  eoFlOrAllAtomBinOp( eoBinOp<AtomType>& _op, float _rate = 1.0):
    op(_op), rate( _rate ) {}

  /** applies Atom crossover to ALL components with given rate */
  bool operator()(EOT & _eo1, const EOT & _eo2)
  {
    if (_eo1.size() != _eo2.size())
      {
        string s = "Operand size don't match in " + className();
        throw runtime_error(s);
      }
    bool changed = false;
    for ( unsigned i = 0; i < _eo1.size(); i++ ) {
      if ( rng.flip( rate ) ) {
        bool changedHere = op( _eo1[i], _eo2[i] );
        changed |= changedHere;
      }
    }
    return changed;
  }

  /** inherited className()*/
  virtual string className() const { return "eoFlOrAllAtomBinOp"; }

private:
  double rate;
  eoBinOp<AtomType> & op;
};

//////////////////////////////////////////////////////////////////////
//                 eoFlOrKAtomBinOp
//////////////////////////////////////////////////////////////////////
/** Bin Crossover using an Atom Crossover
 *  that is applied to a FIXED NB of components
 */
template <class EOT>
class eoFlOrKAtomBinOp : public eoBinOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;

  /** default ctor: requires an Atom BinOp and an unsigned */
  eoFlOrAtomBinOp( eoBinOp<AtomType>& _op, unsigned _k = 1):
    op(_op), k( _k ) {}

  /** applies the Atom BinOp to some components */
  bool operator()(EOT & _eo1, const EOT & _eo2)
  {
    if (_eo1.size() != _eo2.size())
      {
        string s = "Operand size don't match in " + className();
        throw runtime_error(s);
      }

    bool changed = false;
    for ( unsigned i = 0; i < k; i++ ) //! @todo check that we don't do twice the same
      {
        unsigned where = eo::rng.random(_eo1.size());
        bool changedHere = op( _eo1[where], _eo2[where] );
        changed |= changedHere;
      }
    return changed;
  }

  /** inherited className()*/
  virtual string className() const { return "eoFlOrKAtomBinOp"; }

private:
  unsigned k;
  eoBinOp<AtomType> & op;
};


//////////////////////////////////////////////////////////////////////
//                        eoFlOrUniformBinOp
//////////////////////////////////////////////////////////////////////

/** The uniform crossover - exchanges atoms uniformly ! */
template <class EOT>
class eoFlOrUniformBinOp : public eoBinOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;

  /** default ctor: requires a rate - 0.5 by default */
  eoFlOrUniformBinOp(double _rate=0.5) : eoBinOp<EOT>(_size),
    rate(_rate) {}

  /** excahnges atoms at given rate */
  bool operator()(EOT & _eo1, const EOT & _eo2)
  {
    unsigned i;
    Atom tmp;
    if (_eo1.size() != _eo2.size())
      {
        string s = "Operand size don't match in " + className();
        throw runtime_error(s);
  }
    bool hasChanged = false;
    for (unsigned i=0; i<_eo1.size(); i++)
      {
        if ( (_eo1[i]!=_eo2[i]) && (eo::rng.filp(rate)) )
        {
          _eo1[i] = _eo2[i];
          hasChanged = true;
        }
      }
    return hasChanged;
  }

  /** inherited className()*/
  virtual string className() const { return "eoFlOrUniformBinOp"; }

private:
  double rate;
};

//////////////////////////////////////////////////////////////////////
//                        eoFlOr1ptBinOp
//////////////////////////////////////////////////////////////////////

/** The 1pt  crossover (just in case someone wants it some day!) */
template <class EOT>
class eoFlOr1ptBinOp : public eoBinOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;

  /** default ctor: no argument */
  eoVlUniformBinOp() {}

  /** exchanges first and second parts of the vectors of Atoms */
  bool operator()(EOT & _eo1, EOT & _eo2)
  {
    unsigned i;
    Atom tmp;
    if (_eo1.size() != _eo2.size())
      {
        string s = "Operand size don't match in " + className();
        throw runtime_error(s);
  }
    bool hasChanged = false;
    unsigned where = eo::rng.random(_eo1.size()-1);
    for (unsigned i=where+1; i<_eo1.size(); i++)
      {
        if ( (_eo1[i]!=_eo2[i]) )
        {
          _eo1[i] = _eo2[i];
          hasChanged = true;
        }
      }
    return hasChanged;
  }

  /** inherited className()*/
  virtual string className() const { return "eoFlOr1ptBinOp"; }

};

/** @} */

#endif
