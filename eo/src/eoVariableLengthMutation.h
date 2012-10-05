// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoVariableLengthMutation.h
// (c) Marc Schoenauer 1999 - Maarten Keijzer 2000
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

    Contact: Marc.Schoenauer@polytechnique.fr
             mkeijzer@cs.vu.nl
 */
//-----------------------------------------------------------------------------

#ifndef _eoVariableLengthMutation_h
#define _eoVariableLengthMutation_h

#include <eoFunctor.h>
#include <eoOp.h>
#include <eoInit.h>

/**
  Base classes for generic mutations on variable length chromosomes.
  Contains addition and deletion of a gene

  The generic mutations that apply a gene-level mutation to some genes
  dont't modify the length, and so are NOT specific to variable-length
  Hence they are in file eoFlOr MonOp.h file (FixedLengthOrdered mutations)
*/

/* @addtogroup Variators
 * @{
 */

/** Addition of a gene
    Is inserted at a random position - so can be applied to both
    order-dependent and order-independent
 */
template <class EOT>
class eoVlAddMutation : public eoMonOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;

  /** default ctor

   * @param _nMax      max number of atoms
   * @param _atomInit an Atom initializer
   */
  eoVlAddMutation(unsigned _nMax, eoInit<AtomType> & _atomInit) :
    nMax(_nMax), atomInit(_atomInit) {}

  /** operator: actually adds an Atom */
  bool operator()(EOT & _eo)
  {
    if (_eo.size() >= nMax)
      return false;                // unmodifed
    AtomType atom;
    atomInit(atom);
    unsigned pos = rng.random(_eo.size()+1);
    _eo.insert(_eo.begin()+pos, atom);
    return true;
  }

  /** inherited className */
  virtual std::string className() const { return "eoVlAddMutation"; }

private:
  unsigned nMax;
  eoInit<AtomType> & atomInit;
};


/** A helper class for choosing which site to delete */
template <class EOT>
class eoGeneDelChooser : public eoUF<EOT &, unsigned int>
{
public:
  virtual std::string className() const =0;

};

/** Uniform choice of gene to delete */
template <class EOT>
class eoUniformGeneChooser: public eoGeneDelChooser<EOT>
{
public:
    eoUniformGeneChooser(){}
    unsigned operator()(EOT & _eo)
    {
        return eo::rng.random(_eo.size());
    }
  virtual std::string className() const { return "eoUniformGeneChooser"; }
};

/** Deletion of a gene
    By default at a random position, but a "chooser" can be specified
    can of course be applied to both order-dependent and order-independent
 */
template <class EOT>
class eoVlDelMutation : public eoMonOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;

  /** ctor with an external gene chooser

   * @param _nMin      min number of atoms to leave in the individual
   * @param _chooser   an eoGeneCHooser to choose which one to delete
   */
  eoVlDelMutation(unsigned _nMin, eoGeneDelChooser<EOT> & _chooser) :
    nMin(_nMin), uChooser(), chooser(_chooser) {}

  /** ctor with uniform gene chooser - the default

   * @param _nMin      min number of atoms to leave in the individual
   */
  eoVlDelMutation(unsigned _nMin) :
    nMin(_nMin), uChooser(), chooser(uChooser) {}

  /** Do the job (delete one gene)
   * @param _eo  the EO to mutate
   */
  bool operator()(EOT & _eo)
  {
    if (_eo.size() <= nMin)
      return false;                // unmodifed
    unsigned pos = chooser(_eo);
    _eo.erase(_eo.begin()+pos);
    return true;
  }

  virtual std::string className() const
  {
    std::ostringstream os;
    os << "eoVlDelMutation("<<chooser.className() << ")";
    return os.str();
  }

private:
    unsigned nMin;
    eoUniformGeneChooser<EOT> uChooser;
    eoGeneDelChooser<EOT> & chooser;
};

/** @} */
#endif
