/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

-----------------------------------------------------------------------------
    eoData.h
        Some numeric limits and types and things like that; with #ifdefs to keep
	compatibility
    (c) GeNeura Team & Maarten Keijzer, 1998, 1999, 2000
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

#ifndef EODATA_H
#define EODATA_H

//-----------------------------------------------------------------------------

#include <vector>           // std::vector
#include <set>              // set
#include <string>           // std::string


#ifdef _MSC_VER 
	#include <limits>    // MAXDOUBLE 
	#define  MAXFLOAT  numeric_limits<float>::max()
	#define  MINFLOAT  numeric_limits<float>::min()
	#define  MAXDOUBLE  numeric_limits<double>::max() 
	#define  MAXINT numeric_limits<int>::max() 
#else	
        #include <float.h>
        #include <limits.h>
#endif

#if !defined(_WIN32) && !defined(__CYGWIN__) && !(defined(__APPLE__) || defined(MACOSX)) && !defined(__FreeBSD__) 
	#include <values.h>
#endif

// for cygwin and windows (and possibly MacOsX)
#ifndef MINFLOAT
     #define MINFLOAT ( (float)1e-127 )
#endif
#ifndef MAXFLOAT
     #define MAXFLOAT ( (float)1e127 )
#endif
#ifndef MAXINT
	#define MAXINT 2147483647
#endif
#ifndef MAXDOUBLE
 	#define MAXDOUBLE  (double)1.79769313486231570e+308
#endif	

#ifndef _MSC_VER
#include <math.h>
#define _isnan isnan
#endif

//-----------------------------------------------------------------------------
// some defines to make things easier to get at first sight

// tuning the amount of output using a boolean argument: 
// true should always mean more output
#define eo_verbose true
#define eo_no_verbose false
// to be used in selection / replacement procedures to indicate whether 
// the argument (rate, a double) shoudl be treated as a rate (number=rate*popSize)
// or as an absolute integer (number=rate regardless of popsize).
// the default value shoudl ALWAYS be true (eo_as_a_rate).
//
// this construct is mandatory because in some cases you might not know the 
// population size that will enter the replacement for instance - so you 
// cannot simply have a pre-computed (double) rate of 1/popSize
#define eo_is_a_rate true
#define eo_is_an_integer false

#endif

