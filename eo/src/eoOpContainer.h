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
 */
template <class EOT>
class eoOpContainer : public eoGenOp<EOT>
{
  public :
  /** Ctor: nothing much to do */
  eoOpContainer() : max_to_produce(0) {}

  /** Dtor: delete all the GenOps created when wrapping simple ops
   */
  virtual ~eoOpContainer(void) 
  { 
    for (unsigned i = 0; i < owned_genops.size(); ++i) 
      delete owned_genops[i]; 
  }

  /** for memory management (doesn't have to be very precise */
  virtual unsigned max_production(void)
  {
    return max_to_produce;
  }

  /** wraps then add a simple eoMonOp */
  void add(eoMonOp<EOT>& _op, double _rate)
  {
    owned_genops.push_back(new eoMonGenOp<EOT>(_op));
    ops.push_back(owned_genops.back());
    rates.push_back(_rate);

    max_to_produce = max(max_to_produce,unsigned(1));
  }

  /** wraps then add a simple eoBinOp 
   *  First case, no selector
   */
  void add(eoBinOp<EOT>& _op, double _rate)
  {
    owned_genops.push_back(new eoBinGenOp<EOT>(_op));
    ops.push_back(owned_genops.back());
    rates.push_back(_rate);

    max_to_produce = max(max_to_produce,unsigned(1));
  }

  /** wraps then add a simple eoBinOp 
   *  Second case: a sepecific selector
   */
  void add(eoBinOp<EOT>& _op, eoSelectOne<EOT> & _sel, double _rate)
  {
    owned_genops.push_back(new eoSelBinGenOp<EOT>(_op, _sel));
    ops.push_back(owned_genops.back());
    rates.push_back(_rate);

    max_to_produce = max(max_to_produce,unsigned(1));
  }

  /** wraps then add a simple eoQuadOp */
  void add(eoQuadOp<EOT>& _op, double _rate)
  {
    owned_genops.push_back(new eoQuadGenOp<EOT>(_op));
    ops.push_back(owned_genops.back());
    rates.push_back(_rate);

    max_to_produce = max(max_to_produce,unsigned(2));
}

  /** can add any GenOp */
  void add(eoGenOp<EOT>& _op, double _rate)
  {
    ops.push_back(&_op);
    rates.push_back(_rate);

    max_to_produce = max(max_to_produce,_op.max_production());
  }

  virtual string className() = 0;

  protected :

  vector<double> rates;
  vector<eoGenOp<EOT>*> ops;

  private :
  vector<eoGenOp<EOT>*> owned_genops;
  unsigned max_to_produce;
};

/** Sequential selection:
 *  note the mark, rewind, unmark cycle
 *  here operators are repeatedly applied on the same individual(s)
 *  not all too elegant, but it sort of works...
 */
template <class EOT>
class eoSequentialOp : public eoOpContainer<EOT>
{
  public :
  typedef unsigned position_type;


  void apply(eoPopulator<EOT>& _pop)
  {
     position_type pos = _pop.tellp();

     for (size_t i = 0; i < rates.size(); ++i)
     {
        _pop.seekp(pos);

        do
        {
          if (eo::rng.flip(rates[i]))
          {
	    //            try
	    //            {
	      // apply it to all the guys in the todo list
              (*ops[i])(_pop);
	      //            }
            // check for out of individuals and do nothing with that...
	      //            catch(eoPopulator<EOT>::OutOfIndividuals&)
	      //	      {
	      //		cout << "Warning: not enough individuals to handle\n";
	      //		return ;
	      //	      }
          }

          if (!_pop.exhausted())
            ++_pop;
        }
        while (!_pop.exhausted());
     }
  }
  virtual string className() {return "SequentialOp";}

  private :

  vector<size_t> to_apply;
  vector<size_t> production;
};


/** The proportinoal verions: easy! */
template <class EOT>
class eoProportionalOp : public eoOpContainer<EOT>
{
    public :

    void apply(eoPopulator<EOT>& _pop)
    {
      unsigned i = eo::rng.roulette_wheel(rates);

      try
      {
        (*ops[i])(_pop);
      }
      catch(eoPopulator<EOT>::OutOfIndividuals&)
      {}
    }
  virtual string className() {return "ProportionalOp";}
};


#endif

