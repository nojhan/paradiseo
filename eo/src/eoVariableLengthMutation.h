// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoVariableLengthMutation.h
// (c) GeNeura Team, 2000 - EEAAX 1999 - Maarten Keijzer 2000
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
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
    CVS Info: $Date: 2001-07-11 06:26:11 $ $Version$ $Author: evomarc $ 
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

   * @param nMax      max number of atoms
   * @param _atomInit an Atom initializer
   */
  eoVlAddMutation(unsigned _nMax, eoInit<AtomType> & _atomInit) :
    nMax(_nMax), atomInit(_atomInit) {}

  bool operator()(EOT & _eo)
  {
    if (_eo.size() >= nMax)
      return false;		   // unmodifed
    AtomType atom;
    atomInit(atom);
    unsigned pos = rng.random(_eo.size()+1);
    _eo.insert(_eo.begin()+pos, atom);
    return true;
  }
private:
  unsigned nMax;
  eoInit<AtomType> & atomInit;
};

/** A helper class for choosing which site to delete */
template <class EOT>
class eoGeneDelChooser : public eoUF<EOT &, unsigned int>
{};

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

   * @param nMin      min number of atoms to leave in the individual
   * @param _geneChooser an eoGeneCHooser to choose which one to delete
   */
  eoVlDelMutation(unsigned _nMin, eoGeneDelChooser<EOT> & _chooser) :
    nMin(_nMin), uChooser(), chooser(_chooser) {}

  /** ctor with uniform gene chooser

   * @param nMin      min number of atoms to leave in the individual
   */
  eoVlDelMutation(unsigned _nMin) :
    nMin(_nMin), uChooser(), chooser(uChooser) {}

  /** Do the job (delete one gene)
   * @param _eo  the EO to mutate
   */
  bool operator()(EOT & _eo)
  {
    if (_eo.size() <= nMin)
      return false;		   // unmodifed
    unsigned pos = chooser(_eo);
    _eo.erase(_eo.begin()+pos);
    return true;
  }
private:
    unsigned nMin;
    eoUniformGeneChooser<EOT> uChooser;
    eoGeneDelChooser<EOT> & chooser;
};



#endif
