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

#include <vector>           // vector
#include <set>              // set
#include <string>           // string

using namespace std;


#ifdef _MSC_VER 
	#include <limits>    // MAXDOUBLE 
	#define  MAXFLOAT  numeric_limits<float>::max()
	#define  MINFLOAT  numeric_limits<float>::min()
	#define  MAXDOUBLE  numeric_limits<double>::max() 
	#define  MAXINT numeric_limits<int>::max() 
#else	
        #include <float.h>
        #include <limits.h>
#ifndef _WIN32    // should be the define for UN*X flavours: _POSIX??
        #include <values.h>
#endif
	#ifndef MAXFLOAT
		#define  MAXFLOAT (float)1e127
		#define  MAXDOUBLE  (double)1.79769313486231570e+308
		#define  MAXINT 2147483647
	#endif
#endif

#ifndef _MSC_VER
#include <math.h>
#define _isnan isnan
#endif

//-----------------------------------------------------------------------------

#endif EODATA_H
