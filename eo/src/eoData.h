//-----------------------------------------------------------------------------
// eoData.h
//-----------------------------------------------------------------------------

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
        #include <values.h>
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
