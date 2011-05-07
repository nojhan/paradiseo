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

    Contact: mak@dhi.dk
             Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

#ifndef _eoGenOp_H
#define _eoGenOp_H

#include <eoOp.h>
#include <eoPopulator.h>
#include <eoFunctorStore.h>
#include <assert.h>

/** @name General variation operators

a class that allows to use i->j operators for any i and j
thanks to the friend class eoPopulator

@author Maarten Keijzer
@version 0.0

@ingroup Core
@ingroup Variators
*/

//@{

/** The base class for General Operators
Subclass this operator is you want to define an operator that falls
outside of the eoMonOp, eoBinOp, eoQuadOp classification. The argument
the operator will receive is an eoPopulator, which is a wrapper around
the original population, is an instantiation of the next population and
has often a selection function embedded in it to select new individuals.

Note that the actual work is performed in the apply function.
AND that the apply function is responsible for invalidating
the object if necessary
 */
template <class EOT>
class eoGenOp : public eoOp<EOT>, public eoUF<eoPopulator<EOT> &, void>
{
  public :
  /// Ctor that honors its superclass
  eoGenOp(): eoOp<EOT>( eoOp<EOT>::general ) {}

  /** Max production is used to reserve space for all elements that are used by the operator,
      not setting it properly can result in a crash
  */
    virtual unsigned max_production(void) = 0;

    virtual std::string className() const = 0;

    void operator()(eoPopulator<EOT>& _pop)
    {
        _pop.reserve( max_production() );
        apply(_pop);
    }

    //protected :
  /** the function that will do the work
   */
    virtual void apply(eoPopulator<EOT>& _pop) = 0;
};
/** @example t-eoGenOp.cpp
 */


/** Wrapper for eoMonOp */
template <class EOT>
class eoMonGenOp : public eoGenOp<EOT>
{
   public:
    eoMonGenOp(eoMonOp<EOT>& _op) : op(_op) {}

    unsigned max_production(void) { return 1; }

    void apply(eoPopulator<EOT>& _it)
    {
      if (op(*_it))
        (*_it).invalidate();  // look how simple

    }
  virtual std::string className() const {return op.className();}
   private :
    eoMonOp<EOT>& op;
};

/** Wrapper for binop: here we use select method of eoPopulator
 *  but we could also have an embedded selector to select the second parent
 */
template <class EOT>
class eoBinGenOp : public eoGenOp<EOT>
{
   public:
    eoBinGenOp(eoBinOp<EOT>& _op) : op(_op) {}

    unsigned max_production(void) { return 1; }

  /** do the work: get 2 individuals from the population, modifies
      only one (it's a eoBinOp)
      */
    void apply(eoPopulator<EOT>& _pop)
    {
      EOT& a = *_pop;
      const EOT& b = _pop.select();

      if (op(a, b))
        a.invalidate();
    }
  virtual std::string className() const {return op.className();}

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
      if (op(*_pop, sel(_pop.source())))
        (*_pop).invalidate();
    }
  virtual std::string className() const {return op.className();}

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


      if(op(a, b))
      {
        a.invalidate();
        b.invalidate();
      }

   }
  virtual std::string className() const {return op.className();}

   private :
    eoQuadOp<EOT>& op;
};

    /**
    Factory function for automagically creating references to an
    eoGenOp object. Useful when you are too lazy to figure out
    which wrapper belongs to which operator. The memory allocated
    in the wrapper will be stored in a eoFunctorStore (eoState derives from this).
    Therefore the memory will only be freed when the eoFunctorStore is deleted.
    Make very sure that you are not using these wrappers after this happens.

    You can use this function 'wrap_op' in the following way. Suppose you've
    created an eoQuadOp<EOT> called my_quad, and you want to feed it to an eoTransform
    derived class that expects an eoGenOp<EOT>. If you have an eoState lying around
    (which is generally a good idea) you can say:

    eoDerivedTransform<EOT> trans(eoGenOp<EOT>::wrap_op(my_quad, state), ...);

    And as long as your state is not destroyed (by going out of scope for example,
    your 'trans' functor will be usefull.

    As a final note, you can also enter an eoGenOp as the argument. It will
    not allocate memory then. This to make it even easier to use the wrap_op function.
    For an example of how this is used, check the eoOpContainer class.

    @see eoOpContainer
    */
    template <class EOT>
    eoGenOp<EOT>& wrap_op(eoOp<EOT>& _op, eoFunctorStore& _store)
    {
      switch(_op.getType())
      {
        case eoOp<EOT>::unary     : return _store.storeFunctor(new eoMonGenOp<EOT>(static_cast<eoMonOp<EOT>&>(_op)));
        case eoOp<EOT>::binary    : return _store.storeFunctor(new eoBinGenOp<EOT>(static_cast<eoBinOp<EOT>&>(_op)));
        case eoOp<EOT>::quadratic : return _store.storeFunctor(new eoQuadGenOp<EOT>(static_cast<eoQuadOp<EOT>&>(_op)));
        case eoOp<EOT>::general   : return static_cast<eoGenOp<EOT>&>(_op);
      }

      assert(false);
      return static_cast<eoGenOp<EOT>&>(_op);
    }

#endif

//@}
