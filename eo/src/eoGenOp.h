// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGenOp.h 
// (c) Maarten Keijzer and Marc Schoenauer, 2001
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

    Contact: mkeijzer@dhi.dk
             Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

#ifndef _eoGenOp_H
#define _eoGenOp_H

#include <eoOp.h>
#include <eoPopulator.h>


/** @name General variation operators

a class that allows to use i->j operators for any i and j
thanks to the friend class eoPopulator

@author Maarten Keijzer
@version 0.0
*/


/** The base class for General Operators
 */
template <class EOT>
class eoGenOp : public eoOp<EOT>, public eoUF<eoPopulator<EOT> &, void>
{
  public :
  /// Ctor that honors its superclass
  eoGenOp(): eoOp<EOT>( eoOp<EOT>::general ) {}  

    virtual unsigned max_production(void) = 0;
  virtual string className() = 0;
    void operator()(eoPopulator<EOT>& _pop)
    {
      _pop.reserve(max_production());
      apply(_pop);
    }

  protected :
  /** the function that will do the work
   */
    virtual void apply(eoPopulator<EOT>& _pop) = 0;
};


/** Wrapper for eoMonOp */
template <class EOT>
class eoMonGenOp : public eoGenOp<EOT>
{
   public:
    eoMonGenOp(eoMonOp<EOT>& _op) : op(_op) {}

    unsigned max_production(void) { return 1; }

    void apply(eoPopulator<EOT>& _it)
    {
      op(*_it);  // look how simple

    }
  string className() {return op.className();}
   private :
    eoMonOp<EOT>& op;
};

/** Wrapper for binop: here we use erase method of eoPopulator
 *  but we could also have an embedded selector to select the second parent
 */
template <class EOT>
class eoBinGenOp : public eoGenOp<EOT>
{
   public:
    eoBinGenOp(eoBinOp<EOT>& _op) : op(_op) {}

    unsigned max_production(void) { return 1; }

  /** do the work: get 2 individuals from the population, modifies
      only one (it's a eoBinOp) and erases the non-midified one
      */
    void apply(eoPopulator<EOT>& _pop)
    {
      EOT& a = *_pop;
      EOT& b = *++_pop;
      op(a, b);
      _pop.erase();
    }
  string className() {return op.className();}

   private :
    eoBinOp<EOT>& op;
};

/** wrapper for eoBinOp with a selector */
template <class EOT>
class eoSelBinGenOp : public eoGenOp<EOT>
{
   public:
    eoSelBinGenOp(eoBinOp<EOT>& _op, eoSelectOne<EOT>& _sel) : 
      op(_op), sel(_sel) {}
 
    unsigned max_production(void) { return 1; }
 
    void apply(eoPopulator<EOT>& _pop)
    { // _pop.source() gets the original population, an eoVecOp can make use of this as well
      op(*_pop, sel(_pop.source()));
    }
  string className() {return op.className();}
 
   private :
    eoBinOp<EOT>& op;
    eoSelectOne<EOT>& sel;
};


/** Wrapper for quadop: easy as pie
 */
template <class EOT>
class eoQuadGenOp : public eoGenOp<EOT>
{
   public:
    eoQuadGenOp(eoQuadOp<EOT>& _op) : op(_op) {}

    unsigned max_production(void) { return 2; }

    void apply(eoPopulator<EOT>& _pop)
    {
      EOT& a = *_pop;
      EOT& b = *++_pop;

      op(a, b);
   }
  string className() {return op.className();}

   private :
    eoQuadOp<EOT>& op;
};


#endif

