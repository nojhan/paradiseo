/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoWrappedOps.h
    Derived from the General genetic operator, which can be used to wrap any unary or binary
    operator. File also contains the eoCombinedOp, needed by the eoSequentialGOpSelector

 (c) Maarten Keijzer (mak@dhi.dk) and GeNeura Team, 1999, 2000
 
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

#ifndef eoWrappedOps_h
#define eoWrappedOps_h

//-----------------------------------------------------------------------------

#include <eoOp.h>          // eoOp, eoMonOp, eoBinOp
#include <utils/eoRNG.h>

using namespace std;

/// Wraps monary operators
template <class EOT>
class eoWrappedMonOp : public eoGeneralOp<EOT>
{
public :
  ///
  eoWrappedMonOp(eoMonOp<EOT>& _op) : eoGeneralOp<EOT>(), op(_op) {};

  ///
  virtual ~eoWrappedMonOp() {}

  /// Instantiates the abstract method
  void operator()( eoIndiSelector<EOT>& _in, 
			   eoInserter<EOT>& _out) 
  {
    EOT result = _in();
    op( result );
    _out(result);
  }
  
private :
  eoMonOp<EOT>& op;
};


/// Wraps binary operators
template <class EOT>
class eoWrappedBinOp : public eoGeneralOp<EOT>
{
public :
  ///
  eoWrappedBinOp(eoBinOp<EOT>& _op) : eoGeneralOp<EOT>(), op(_op) {}

  ///
  virtual ~eoWrappedBinOp() {}

  /// Instantiates the abstract method. EOT should have copy ctor.
  void operator()(eoIndiSelector<EOT>& _in, 
			   eoInserter<EOT>& _out) 
  {
    EOT out1 = _in();
    const EOT& out2 = _in();
    op(out1, out2);
    _out(out1);
  }

private :
	eoBinOp<EOT>& op;
};

/// Wraps Quadratic operators
template <class EOT>
class eoWrappedQuadraticOp : public eoGeneralOp<EOT>
{
public :
  ///
  eoWrappedQuadraticOp(eoQuadraticOp<EOT>& _op) : eoGeneralOp<EOT>(), op(_op) {}

  ///
  virtual ~eoWrappedQuadraticOp() {}

  /// Instantiates the abstract method. EOT should have copy ctor.
  void operator()(eoIndiSelector<EOT>& _in, 
			   eoInserter<EOT>& _out) 
  {
    EOT out1 = _in();
    EOT out2 = _in();
    op(out1, out2);
    _out(out1)(out2);
  }
  
private :
	eoQuadraticOp<EOT>& op;
};

#include <eoBackInserter.h>

template <class EOT> 
class eoCombinedOp : public eoGeneralOp<EOT>
{
    public :
    eoCombinedOp& bind(const std::vector<eoGeneralOp<EOT>*>& _ops, const std::vector<float>& _rates)
    { 
        ops = &_ops;
        rates = &_rates;
        return *this;
    }

    class eoDelayedSelector : public eoIndiSelector<EOT>
    {
    public :
        eoDelayedSelector(eoIndiSelector<EOT>& _select, const eoPop<EOT>& _pop) : select(_select), pop(_pop), it(pop.begin()) {}
    
        unsigned size()                 const { return select.size();}
        const EOT& operator[](size_t i) const { return select[i]; }

        /// will first dispense all previously selected individuals before returning new ones
        const EOT& operator()(void)
        {
            if (it == pop.end())
            {
                return select();
            }
            // else
            return *it++;
        }

        eoPop<EOT>::const_iterator get_it(void) const { return it; }
    private :
        eoIndiSelector<EOT>& select;
        const eoPop<EOT>& pop;
        eoPop<EOT>::const_iterator it;
    };

  /** Applies all ops in the combined op
        It first applies the 
  */
  void operator()( eoIndiSelector<EOT>& _in, 
			   eoInserter<EOT>& _out ) 
  {
        eoPop<EOT> intermediate;
        eoPop<EOT> next;
        unsigned i;

        for (i = 0; i < ops->size(); ++i)
        {
            eoDelayedSelector delay(_in, intermediate);
            inserter.bind(next);

            unsigned counter = 0;

            // apply operators until we have as many outputs as inputs
            do
            {
                if (rng.flip(rates->operator[](i))) // should this flip be here?
                    (*ops->operator[](i))(delay, inserter);

                counter++;
                if (counter > 1000) 
                {
                    throw logic_error("eoCombinedOp: no termination after 1000 tries, did you forget to insert individuals in your eoGeneralOp?");
                }
            }
            while (next.size() < intermediate.size());

            intermediate.swap(next);
            next.resize(0);
        }
        
        // after last swap, results can be found in intermediate
        for (i = 0; i < intermediate.size(); ++i)
            _out(intermediate[i]);
  }

    private :
        const std::vector<eoGeneralOp<EOT>*>* ops;
        const std::vector<float>* rates;
        eoBackInserter<EOT> inserter;
};

#endif 

