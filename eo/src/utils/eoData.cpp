// Copyright (C) 2005 Jochen Küpper <jochen@fhi-berlin.mpg.de>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// this program; see the file License. if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.


#include "eoData.h"

#ifdef HAVE_NUMERIC_LIMITS
int MAXINT = numeric_limits<int>::max();
#else
#include <limits.h>
int MAXINT = INT_MAX;
#endif


// Local Variables:
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
