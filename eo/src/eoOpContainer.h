// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoOpContainer.h
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

#ifndef _eoOpContainer_H
#define _eoOpContainer_H

#include <eoGenOp.h>

/** eoOpContainer is a base class for the sequential and proportional selectors
 *  It takes care of wrapping the other operators,
 *  and deleting stuff that it has allocated
 *
 * Warning: all operators are added together with a rate (double)
 *          However, the meaning of this rate will be different in
 *          the differnet instances of eoOpContainer:
 *          an ***absolute*** probability in the sequential version, and
 *          a ***relative*** weight in the proportional version
 *
 *  @ingroup Combination
 */
template <class EOT>
class eoOpContainer : public eoGenOp<EOT>
{
  public :
  /** Ctor: nothing much to do */
  eoOpContainer() : max_to_produce(0) {}

  /** Dtor: delete all the GenOps created when wrapping simple ops
   */
  virtual ~eoOpContainer(void) {}

  /** for memory management (doesn't have to be very precise */
  virtual unsigned max_production(void)
  {
    return max_to_produce;
  }

  /**
  Add an operator to the container, also give it a rate

  (sidenote, it's much less hairy since I added the wrap_op is used)
  */
  void add(eoOp<EOT>& _op, double _rate)
  {
    ops.push_back(&wrap_op<EOT>(_op, store));
    rates.push_back(_rate);
    max_to_produce = std::max(max_to_produce,ops.back()->max_production());
  }

  virtual std::string className() const = 0;

  protected :

  std::vector<double> rates;
  std::vector<eoGenOp<EOT>*> ops;

  private :
  eoFunctorStore store;
  unsigned max_to_produce;
};

/** Sequential selection:
 *  note the mark, rewind, unmark cycle
 *  here operators are repeatedly applied on the same individual(s)
 *  not all too elegant, but it sort of works...
 *
 *  @ingroup Combination
 */
template <class EOT>
class eoSequentialOp : public eoOpContainer<EOT>
{
public:

    using eoOpContainer<EOT>::ops;
    using eoOpContainer<EOT>::rates;

    typedef unsigned position_type;


    void apply(eoPopulator<EOT>& _pop) {
        _pop.reserve( this->max_production() );

        position_type pos = _pop.tellp();
        for (size_t i = 0; i < rates.size(); ++i) {
            _pop.seekp(pos);
            do {
                if (eo::rng.flip(rates[i])) {
                    //            try
                    //            {
                    // apply it to all the guys in the todo std::list

                    //(*ops[i])(_pop);

                    ops[i]->apply(_pop);

                    //            }
                    // check for out of individuals and do nothing with that...
                    //            catch(eoPopulator<EOT>::OutOfIndividuals&)
                    //        {
                    //		std::cout << "Warning: not enough individuals to handle\n";
                    //		return ;
                    //        }
                }

                if (!_pop.exhausted())
                    ++_pop;
            }
            while (!_pop.exhausted());
        }
    }
    virtual std::string className() const {return "SequentialOp";}

private:

    std::vector<size_t> to_apply;
    std::vector<size_t> production;
};



/** The proportional versions: easy! */
template <class EOT>
class eoProportionalOp : public eoOpContainer<EOT>
{
public:

    using eoOpContainer< EOT >::ops;
    using eoOpContainer< EOT >::rates;

    void apply(eoPopulator<EOT>& _pop)
    {
      unsigned i = eo::rng.roulette_wheel(rates);

      try
      {
        (*ops[i])(_pop);
        ++_pop;
      }
      catch( typename eoPopulator<EOT>::OutOfIndividuals&)
      {}
    }
  virtual std::string className() const {return "ProportionalOp";}
};


#endif
