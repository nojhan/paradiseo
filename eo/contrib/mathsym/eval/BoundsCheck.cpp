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

#include <vector>

#include "BoundsCheck.h"
#include <Sym.h>
#include <FunDef.h>
#include <sstream>

using namespace std;

class IntervalBoundsCheckImpl {
    public :
    vector<Interval> bounds;
};

IntervalBoundsCheck::IntervalBoundsCheck(const vector<double>& mins, const vector<double>& maxes) {
    pimpl = new IntervalBoundsCheckImpl;
    vector<Interval>& b = pimpl->bounds;

    b.resize( mins.size());
   
    for (unsigned i = 0; i < b.size(); ++i) {
	b[i] = Interval(mins[i], maxes[i]);
    }
    
}

IntervalBoundsCheck::~IntervalBoundsCheck() { delete pimpl; }
IntervalBoundsCheck::IntervalBoundsCheck(const IntervalBoundsCheck& that) { pimpl = new IntervalBoundsCheckImpl(*that.pimpl); }
IntervalBoundsCheck& IntervalBoundsCheck::operator=(const IntervalBoundsCheck& that)   { *pimpl = *that.pimpl; return *this; }

bool IntervalBoundsCheck::in_bounds(const Sym& sym) const {
    Interval bounds; 
    
    try {
	bounds = eval(sym, pimpl->bounds);
	if (!valid(bounds)) return false;
    } catch (interval_error) {
	return false;
    }
    return true;
}

std::string IntervalBoundsCheck::get_bounds(const Sym& sym) const {
    
    try {
	Interval bounds = eval(sym, pimpl->bounds);
	if (!valid(bounds)) return "err";
	ostringstream os;
	os << bounds;
	return os.str();
    } catch (interval_error) {
	return "err";
    }
}


std::pair<double, double> IntervalBoundsCheck::calc_bounds(const Sym& sym) const {

    Interval bounds = eval(sym, pimpl->bounds);
    return make_pair(bounds.lower(), bounds.upper());
}
	

