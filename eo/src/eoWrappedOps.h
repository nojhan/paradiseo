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
  eoWrappedMonOp(const eoMonOp<EOT>& _op) : eoGeneralOp<EOT>(), op(_op) {};

  ///
  virtual ~eoWrappedMonOp() {}

  /// Instantiates the abstract method
  void operator()( eoIndiSelector<EOT>& _in, 
			   eoInserter<EOT>& _out) const {
    EOT result = _in();
    op( result );
    _out(result);
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
  eoWrappedBinOp(const eoBinOp<EOT>& _op) : eoGeneralOp<EOT>(), op(_op) {}

  ///
  virtual ~eoWrappedBinOp() {}

  /// Instantiates the abstract method. EOT should have copy ctor.
  void operator()(eoIndiSelector<EOT>& _in, 
			   eoInserter<EOT>& _out) const {
    EOT out1 = _in();
    const EOT& out2 = _in();
    op(out1, out2);
    _out(out1);
  }
  
  ///
  virtual string className() const {return "eoWrappedBinOp";};

private :
	const eoBinOp<EOT>& op;
};

/// Wraps Quadratic operators
template <class EOT>
class eoWrappedQuadraticOp : public eoGeneralOp<EOT>
{
public :
  ///
  eoWrappedQuadraticOp(const eoQuadraticOp<EOT>& _op) : eoGeneralOp<EOT>(), op(_op) {}

  ///
  virtual ~eoWrappedQuadraticOp() {}

  /// Instantiates the abstract method. EOT should have copy ctor.
  void operator()(eoIndiSelector<EOT>& _in, 
			   eoInserter<EOT>& _out) const {
    EOT out1 = _in();
    EOT out2 = _in();
    op(out1, out2);
    _out(out1)(out2);
  }
  
  ///
  virtual string className() const {return "eoWrappedQuadraticOp";};

private :
	const eoQuadraticOp<EOT>& op;
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
  void addOp(eoGeneralOp<EOT>* _op) 
  {
    ops.push_back(_op);
  }


  /// Erases all operators added so far
  void clear(void) {
    ops.resize(0);
  }

  /// Helper class to make sure that stuff that is inserted will be used again with the next operator
  class eoIndiSelectorInserter : public eoIndiSelector<EOT>, public eoInserter<EOT>
  {
  public :
      eoIndiSelectorInserter(eoIndiSelector<EOT>& _in) 
          : eoIndiSelector<EOT>(), eoInserter<EOT>(), in(_in) 
            {}

      size_t          size()      const     { return in.size(); }
      const EOT&    operator[](size_t _n) const { return in[_n]; }
          
      const EOT& operator()(void) 
      {
          if (results.empty())
          {
              return in();
          }
          // else we use the previously inserted individual,
          // an iterator to it is stored in 'results', but the memory
          // is kept by 'intermediate'.

          list<EOT>::iterator it = *results.begin();
          results.pop_front();
          return *it;
      }

      eoInserter<EOT>& operator()(const EOT& _eot)
      {
          intermediate.push_front(_eot);
          results.push_front(intermediate.begin());
          return *this;
      }

      void fill(eoInserter<EOT>& _out)
      {
          typedef list<list<EOT>::iterator>::iterator Iterator;

          for (Iterator it = results.begin(); it != results.end(); ++it)
          {
              _out(**it);
          }

          results.clear();
          intermediate.clear(); // reclaim memory
      }

  private :

      eoIndiSelector<EOT>& in;
      
      // using lists as we need to push and pop a lot
      // 'results' are iterators to the contents of 'intermediate'
      // to prevent copying to and from intermediate...
      list<list<EOT>::iterator> results;        
      list<EOT> intermediate;
  };

  /// Applies all ops in the combined op
  void operator()( eoIndiSelector<EOT>& _in, 
			   eoInserter<EOT>& _out ) const {
    
    eoIndiSelectorInserter in_out(_in);

    for (size_t i = 0; i < ops.size(); ++i)
    {
        (*ops[i])(in_out, in_out);
    }

    in_out.fill(_out);
  }
		
private :
  vector<eoGeneralOp<EOT>* > ops;
};

#endif eoGeneral_h

