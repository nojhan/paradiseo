// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTerm.h
// (c) GeNeura Team, 1999, 2000
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
 */
//-----------------------------------------------------------------------------

#ifndef _EOTERM_H
#define _EOTERM_H

//#include <eoPop.h>

// forward definition for fast(er) compilation
template <class EOT> class eoPop;


/** Termination condition for the genetic algorithm
 * Takes the population as input, returns true for continue,
 * false for termination (although this begs the question why this
 * terminator is not called a continuator)
 * 
 */
template< class EOT>
class eoTerm : public eoObject {
public:

	/// Ctors/dtors
	virtual ~eoTerm() {};

	/** Returns false if the training has to stop, true if it
	 continues \\
	 It is non-const since it might change the internal state
	    of the object, for instance, updating local data.
	*/
	virtual bool operator() ( const eoPop< EOT >& _pop ) = 0 ;
};


/** eoParamTerm, a terminator that compares two statistics and decides whether the
  * algorithm should stop or not. 
*/

#include <utils/eoParam.h>

template<class EOT, class Pred>
class eoBinaryTerm : public eoTerm<EOT>
{
public :

    typedef typename Pred::first_argument_type first_argument_type;
    typedef typename Pred::second_argument_type second_argument_type;

	/// Ctors/dtors
    eoBinaryTerm(first_argument_type& _param1, second_argument_type& _param2) : param1(_param1), param2(_param2), compare() {}

	virtual ~eoBinaryTerm() {};

	/** 
    */
	virtual bool operator() ( const eoPop< EOT >& _pop )
    {
        return compare(param1, param2);
    }
	
  /// Class name.
  virtual string className() const { return "eoStatTerm"; }

private :

    first_argument_type& param1;
    second_argument_type& param2;
    Pred compare;
};

#include <utility>
/** Combined termination condition for the genetic algorithm
 *
 *  The eoCombinedTerm will perform a logical and on all the terminators
 *  contained in it. This means that the terminator will say stop (return false)
 *  when one of the contained terminators says so
 *
 *
 * So now you can say:

    eoTerm1<EOT> term1;
    eoTerm2<EOT> term2;
    eoCombinedTerm<EOT> term3(term1, term2);


template <class EOT>
class eoCombinedTerm : public eoTerm<EOT>, public std::pair<eoTerm<EOT>&, eoTerm<EOT>& >
{
public :

    eoCombinedTerm(const eoTerm<EOT>& _first, const eoTerm<EOT>& _second) : std::pair(_first, _second) {}
    
    ~eoCombinedTerm() {}

    bool operator()(const eoPop<EOT>& _pop)
    {
        if (first(_pop))            
            return second(_pop);

        return false; // quit evolution
    }
};
*/
#endif

