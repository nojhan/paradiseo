/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    -----------------------------------------------------------------------------
    eoGOpSelector.h
      Base class for generalized (n-inputs, n-outputs) operator selectors.
      Includes code and variables that contain operators and rates.
      Also included eoProportionalGOpSelector and eoSequentialGOpSelector, that offer
      a few concrete implementations.

    (c) Maarten Keijzer, GeNeura Team 1998, 1999, 2000
 
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

#ifndef eoGOpSelector_h
#define eoGOpSelector_h

//-----------------------------------------------------------------------------

#include <list>
#include "eoOpSelector.h"
#include "eoWrappedOps.h" // for eoCombinedOp
#include <utils/eoRNG.h>

using namespace std;

/** Base class for alternative selectors, which use the generalized operator
    interface. eoGOpBreeders expects this class */
template<class EOT>
class eoGOpSelector: public eoOpSelector<EOT>, public vector<eoGeneralOp<EOT>*>
{
public:

  typedef eoOpSelector<EOT>::ID ID;

  /// Dtor
  virtual ~eoGOpSelector() {
    for ( list< eoGeneralOp<EOT>* >::iterator i= ownOpList.begin();
	  i != ownOpList.end(); i++ ) {
      delete *i;
    }
  }
  
  /* 
     Add any kind of operator to the operator mix, 
     @param _op      operator, one of eoMonOp, eoBinOp, eoQuadraticOp or eoGeneralOp
     @param _rate    the rate at which it should be applied, it should be a probability
                    
  */
  virtual ID addOp( eoOp<EOT>& _op, float _rate ); 
  // implementation can be found below
  
  /** Retrieve the operator using its integer handle
      @param _id The id number. Should be a valid id, or an exception 
      will be thrown
      @return a reference of the operator corresponding to that id.
  */
  virtual eoOp<EOT>& getOp( ID _id )
  {
    return *operator[](_id);
  }
  
  ///
  virtual void deleteOp( ID _id );
  // implemented below
  
  ///
  virtual eoOp<EOT>* Op()
  {
    return &selectOp();
  }

  /// Select an operator from the operators present here
  virtual eoGeneralOp<EOT>& selectOp() = 0;

  ///
  virtual string className() const { return "eoGOpSelector"; };

  ///
  void printOn(ostream& _os) const {}
  //  _os << className().c_str() << endl; 
  //  for ( unsigned i=0; i!= rates.size(); i++ ) {
  //    _os << *(operator[](i))  << "\t" << rates[i] << endl;
  //  }
  //}


  const vector<float>& getRates(void) const { return rates; }

private :
  vector<float> rates;
  list< eoGeneralOp<EOT>* > ownOpList;
};

/* Implementation of longish functions defined above */

template <class EOT>
inline eoOpSelector<EOT>::ID eoGOpSelector<EOT>::addOp( eoOp<EOT>& _op, 
							float _arg )
{
  eoGeneralOp<EOT>* op; 
  
  if (_op.getType() == eoOp<EOT>::general) 
    {
      op = static_cast<eoGeneralOp<EOT>*>(&_op);
    }
  else
    {
      // if it's not a general op, it's a "old" op; create a wrapped op from it
      // and keep it on a list to delete them afterwards
      // will use auto_ptr when they're readily available
      
      switch(_op.getType())
	{
	case eoOp<EOT>::unary :
	  op=  new eoWrappedMonOp<EOT>(static_cast<eoMonOp<EOT>&>(_op));
	  break;
	case eoOp<EOT>::binary :
	  op =  new eoWrappedBinOp<EOT>(static_cast<eoBinOp<EOT>&>(_op));
	  break;
	case eoOp<EOT>::quadratic :
	  op =  new eoWrappedQuadraticOp<EOT>(static_cast<eoQuadraticOp<EOT>&>(_op));
	  break;
	}
      ownOpList.push_back( op );
    }
  
  // Now 'op' is a general operator, either because '_op' was one or 
  // because we wrapped it in an appropriate wrapper in the code above.

  iterator result = find(begin(), end(), (eoGeneralOp<EOT>*) 0); // search for nullpointer
	  
  if (result == end())
    {
      push_back(op);
      rates.push_back(_arg);
      return size();
    }
  // else
    
  *result = op;
  ID id = result - begin();
  rates[id] = _arg;
  return id;
}

template <class EOT>
inline void  eoGOpSelector<EOT>::deleteOp( ID _id )
{
  eoGeneralOp<EOT>* op = operator[](_id);

  operator[](_id) = 0; 
  rates[_id] = 0.0;
    
  // check oplist and clear it there too.
  
  list< eoGeneralOp<EOT>* >::iterator it = find(ownOpList.begin(), ownOpList.end(), op);

  if(it != ownOpList.end())
    {
      ownOpList.erase(it);
    }
}
  
#endif eoGOpSelector_h

