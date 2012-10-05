/*	    
 *             Copyright (C) 2005 Maarten Keijzer
 *
 *          This program is free software; you can redistribute it and/or modify
 *          it under the terms of version 2 of the GNU General Public License as 
 *          published by the Free Software Foundation. 
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program; if not, write to the Free Software
 *          Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef SYMEVAL_H
#define SYMEVAL_H

#include <Sym.h>
#include <FunDef.h>
#include <ErrorMeasure.h>
#include <BoundsCheck.h>

#include <eoPopEvalFunc.h>

template <class EoType>
class eoSymPopEval : public eoPopEvalFunc<EoType> {

    BoundsCheck&  check;
    ErrorMeasure& measure;
    unsigned size_cap; 
    
    public:

    eoSymPopEval(BoundsCheck& _check, ErrorMeasure& _measure, unsigned _size_cap) :
	check(_check), measure(_measure), size_cap(_size_cap) {}

    /** apparently this thing works on two populations, 
     *
     * In any case, currently only implemented the population wide
     * evaluation version, as that one is much faster. This because the
     * compile going on behind the scenes is much faster when done in one
     * go (and using subtree similarity) then when done on a case by case
     * basis. 
    */
    void operator()(eoPop<EoType>& p1, eoPop<EoType>& p2) {
	
	std::vector<unsigned> unevaluated;
	std::vector<Sym> tmppop;
	
	for (unsigned i = 0; i < p1.size(); ++i) {
	    if (p1[i].invalid()) {

		if (expand_all(p1[i]).size() < size_cap && check.in_bounds(p1[i])) {
		    unevaluated.push_back(i);
		    tmppop.push_back( static_cast<Sym>(p1[i]) );
		} else {
		    p1[i].fitness( measure.worst_performance() );
		}
	    }
	}

	for (unsigned i = 0; i < p2.size(); ++i) {
	    if (p2[i].invalid()) {
		
		if (expand_all(p2[i]).size() < size_cap && check.in_bounds(p2[i])) {
		    
		    unevaluated.push_back(p1.size() + i);
		    tmppop.push_back( static_cast<Sym>(p2[i]) );

		} else {
		    p2[i].fitness( measure.worst_performance() ); // pretty bad error
		}
	    }
	}

	std::vector<ErrorMeasure::result> result = measure.calc_error(tmppop);

	for (unsigned i = 0; i < result.size(); ++i) {
	    unsigned idx = unevaluated[i];

	    if (idx < p1.size()) {
		p1[idx].fitness(result[i].error);
	    } else {
		idx -= p1.size();
		p2[idx].fitness(result[i].error);
	    }
	}
    }

};


#endif
