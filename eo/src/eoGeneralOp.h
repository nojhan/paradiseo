/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoGeneralOp.h
    General genetic operator, which can be used to wrap any unary or binary
    operator
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

#ifndef eoGeneralOp_h
#define eoGeneralOp_h

//-----------------------------------------------------------------------------

#include <vector>          // vector
#include <iterator>
#include <eoUniform.h>     // eoUniform
#include <eoOp.h>          // eoOp, eoMonOp, eoBinOp
#include <eoPop.h>         // eoPop
#include <eoPopOps.h>      // eoTransform
#include <eoOpSelector.h>  // eoOpSelector
#include <list>
#include "eoRNG.h"

using namespace std;

/**
 * eGeneralOp: General genetic operator; for objects used to transform sets
 of EOs
*/
template<class EOT>
class eoGeneralOp: public eoOp<EOT>
{
public:

  /// Ctor that honors its superclass
  eoGeneralOp(): eoOp<EOT>( Nary ) {};

  /// Virtual dtor
  virtual ~eoGeneralOp () {};

  /** Method that really does the stuff. Applies the genetic operator
      to a vector of inputs, and puts results in the output vector */
  virtual void operator()( eoPop<EOT>::iterator _in, 
			   insert_iterator< eoPop<EOT> > _out) const = 0;

  /// Number of inputs
  virtual unsigned nInputs(void) const { return repNInputs;};

  /// Number of output arguments, or arguments that are pushed onto the output vector
  virtual unsigned nOutputs(void) const { return repNOutputs; }; 
  
  virtual string className() const {return "eoGeneralOp";};

protected:
  /// Default ctor; protected so that only derived classes can use it
  eoGeneralOp( unsigned _nInputs = 0, unsigned _nOutputs = 0 )
    : repNInputs( _nInputs), repNOutputs( _nOutputs) {};

  /// change number of inputs
  void setNInputs( unsigned _nInputs) { repNInputs = _nInputs;};
  
  /// change number of outputs
  void setNOutputs( unsigned _nOutputs) { repNOutputs = _nOutputs;};

private:
  unsigned repNInputs;
  unsigned repNOutputs;
};


/// Wraps monary operators
template <class EOT>
class eoWrappedMonOp : public eoGeneralOp<EOT>
{
public :
  ///
  eoWrappedMonOp(const eoMonOp<EOT>& _op) : eoGeneralOp<EOT>( 1, 1), op(_op) {};

  ///
  virtual ~eoWrappedMonOp() {}

  /// Instantiates the abstract method
  void operator()( eoPop<EOT>::iterator _in, 
		   insert_iterator< eoPop< EOT> > _out ) const {
    EOT result = *_in;
    op( result );
    *_out = result;
  }
  
  ///
  virtual string className() const {return "eoWrappedMonOp";};


private :
  const eoMonOp<EOT>& op;
};


/// Wraps binary operators
template <class EOT>
class eoWrappedBinOp : public eoGeneralOp<EOT>
{
public :
  ///
  eoWrappedBinOp(const eoBinOp<EOT>& _op) : eoGeneralOp<EOT>(2, 2), op(_op) {}

  ///
  virtual ~eoWrappedBinOp() {}

  /// Instantiates the abstract method. EOT should have copy ctor.
  void operator()(eoPop<EOT>::iterator _in, 
		  insert_iterator< eoPop< EOT> > _out  ) const {
    EOT out1 = *_in;
    _in++;
    EOT out2 = *_in;
    op(out1, out2);
    *_out++ = out1;
    *_out = out2;
  }
  
  ///
  virtual string className() const {return "eoWrappedBinOp";};

private :
	const eoBinOp<EOT>& op;
};

/// Combines several ops
template <class EOT>
class eoCombinedOp : public eoGeneralOp<EOT>
{
public :
  
  ///
  eoCombinedOp() : eoGeneralOp<EOT>() {}

  ///
  virtual ~eoCombinedOp() {}
 
  /// Adds a new operator to the combined Op
  void addOp(eoGeneralOp<EOT>* _op) {
    ops.push_back(_op);
    unsigned nInputs = nInputs() < _op->nInputs()? _op->nInputs() : nInputs;
    setNInputs( nInputs );
    unsigned nOutputs = nOutputs() < _op->nOutputs()? _op->nOutputs() : nOutputs;
    setNOutputs( nInputs );
  }


  /// Erases all operators added so far
  void clear(void) {
    ops.resize(0);
  }


  /// Applies all ops in the combined op
  void operator()( eoPop<EOT>::iterator _in, 
		   insert_iterator< eoPop< EOT> > _out ) const {
    // used for provisional input and output. Results are put in provOut,
    // and copied back to provIn.
    eoPop<EOT> provIn, provOut;
    insert_iterator< eoPop< EOT> > out = provOut.begin();
    ops[0]( _in, out );
    for ( unsigned i = 1; i < ops.size; i ++ ) {
      copy( provOut.begin(), provOut.end(), provIn.begin() );
      insert_iterator< eoPop< EOT> > in = provIn.begin();
      out = provOut.begin();
      ops[i]( in, out );
    }

    // Copy back to output
    copy( provOut.begin(), provOut.end(), _out );
      
    
  }
		
private :
  vector<eoGeneralOp<EOT>* > ops;
};

#endif eoGeneral_h
