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

#ifndef INTERVAL__H__
#define INTERVAL__H__

#include <boost/numeric/interval.hpp>
#include <iostream>
#include <limits>


typedef boost::numeric::interval_lib::rounded_transc_exact<double> RoundingTransc;
typedef boost::numeric::interval_lib::save_state<RoundingTransc> Rounding;
typedef boost::numeric::interval_lib::checking_base<double> Checking;
typedef boost::numeric::interval_lib::policies<Rounding,Checking> Policy;
typedef boost::numeric::interval<double, Policy> Interval;

struct interval_error{};

inline bool valid(const Interval& val) {
    if (!finite(val.lower()) || !finite(val.upper())) return false;
    
    return val.lower() > -1e10 && val.upper() < 1e10;
}

inline Interval sqrt(const Interval& val) {
    if (val.lower() < 0.0) {
	return Interval::whole();
    }

    return boost::numeric::sqrt(val);
}

inline Interval sqr(const Interval& val) {
    return square(val);
}

inline Interval acos(const Interval& val) {
    if (val.lower() < 1.0 || val.upper() > 1.0) {
	return Interval::whole();
    }

    return boost::numeric::acos(val);
}

inline Interval asin(const Interval& val) {
    if (val.lower() < 1.0 || val.upper() > 1.0) {
	return Interval::whole();
    }

    return boost::numeric::asin(val);
}

inline Interval acosh(const Interval& val) {
    if (val.lower() < 1.0) return Interval::whole();
    return boost::numeric::acosh(val);
}

inline
std::ostream& operator<<(std::ostream& os, const Interval& val) {
        os << '[' << val.lower() << ", " << val.upper() << ']';
	    return os;
}

#ifdef TEST_INTERVAL
#endif

#endif
