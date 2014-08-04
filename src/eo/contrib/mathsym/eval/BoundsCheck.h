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

#ifndef BOUNDS_CHECK_H_
#define BOUNDS_CHECK_H_

#include <string>

class IntervalBoundsCheckImpl;
class Sym;

class BoundsCheck {
    public :
	virtual ~BoundsCheck() {};
	virtual bool in_bounds(const Sym&) const = 0;
	virtual std::string get_bounds(const Sym&) const = 0;
};

// checks if a formula keeps within bounds using interval arithmetic
class IntervalBoundsCheck : public BoundsCheck {

    IntervalBoundsCheckImpl* pimpl;
    
    public:

    IntervalBoundsCheck(const std::vector<double>& minima, const std::vector<double>& maxima);
    ~IntervalBoundsCheck();
    IntervalBoundsCheck(const IntervalBoundsCheck&);
    IntervalBoundsCheck& operator=(const IntervalBoundsCheck&);
    
    bool in_bounds(const Sym&) const;
    std::string get_bounds(const Sym&) const;
    
    std::pair<double, double> calc_bounds(const Sym&) const;
};

class NoBoundsCheck : public BoundsCheck {
    bool in_bounds(const Sym&) const { return false; }
    std::string get_bounds(const Sym&) const { return ""; }
};

#endif


