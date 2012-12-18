/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

-----------------------------------------------------------------------------
    eoData.h

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

#ifndef _MSC_VER
#include <math.h>
#define _isnan isnan
#endif


//-----------------------------------------------------------------------------
// some defines to make things easier to get at first sight

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
