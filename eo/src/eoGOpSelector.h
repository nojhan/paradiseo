/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    -----------------------------------------------------------------------------
    eoGOpSelector.h
      Base class for generalized (n-inputs, n-outputs) operator selectors.
      Includes code and variables that contain operators and rates

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

#include <vector>          // vector
#include <iterator>
#include <eoUniform.h>     // eoUniform
#include <eoGeneralOp.h>          // eoOp, eoMonOp, eoBinOp
#include <eoPop.h>         // eoPop
#include <eoPopOps.h>      // eoTransform
#include <eoOpSelector.h>  // eoOpSelector
#include <list>
#include "eoRNG.h"

using namespace std;

/** Base class for alternative selectors, which use the generalized operator
    interface */

template<class EOT>
class eoGOpSelector: public eoOpSelector<EOT>, public vector<eoGeneralOp<EOT>*>
{
public:

  /// Dtor
  virtual ~eoGOpSelector() {
    for ( list< eoGeneralOp<EOT>* >::iterator i= ownOpList.begin();
	  i != ownOpList.begin(); i ++ ) {
      delete *i;
    }
  }
  
  /// Add any kind of operator to the operator mix, with an argument
  virtual ID addOp( eoOp<EOT>& _op, float _arg ) {
    eoGeneralOp<EOT>* op = dynamic_cast<eoGeneralOp<EOT>*>(&_op);

    // if it's not a general op, it's a "old" op; create a wrapped op from it
    // and keep it on a list to delete them afterwards
    // will use auto_ptr when they're readily available
    if (op == 0) {
	switch(_op.readArity())
	  {
	  case unary :
	    op=  new eoWrappedMonOp<EOT>(static_cast<eoMonOp<EOT>&>(_op));
	    break;
	  case binary :
	    op =  new eoWrappedBinOp<EOT>(static_cast<eoBinOp<EOT>&>(_op));
	    break;
	  }
	ownOpList.push_back( op );
      }

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
  
  /** Retrieve the operator using its integer handle
      @param _id The id number. Should be a valid id, or an exception 
                 will be thrown
      @return a reference of the operator corresponding to that id.
  */
  virtual const eoOp<EOT>& getOp( ID _id )
  {
	  return *operator[](_id);
  }
  
  ///
  virtual void deleteOp( ID _id )
  {
	  operator[](_id) = 0; // TODO, check oplist and clear it there too.
	  rates[_id] = 0.0;
  }
  
  ///
  virtual eoOp<EOT>* Op()
  {
	  return &selectOp();
  }

  ///
  virtual eoGeneralOp<EOT>& selectOp() = 0;

  ///
  virtual string className() const { return "eoGOpSelector"; };

  ///
  void printOn(ostream& _os) const {
    _os << className() << endl; 
    for ( unsigned i=0; i!= rates.size(); i++ ) {
      _os << *(operator[](i))  << "\t" << rates[i] << endl;
    }
  }


protected :
  vector<float> rates;
  list< eoGeneralOp<EOT>* > ownOpList;
};

#endif eoGOpSelector_h
