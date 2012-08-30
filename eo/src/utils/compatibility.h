/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  compatibility.h
      File to store some compiler specific stuff in. Currently handles, or
      least tries to handle the min() max() problems when using MSVC


 (c) Maarten Keijzer (mak@dhi.dk) and GeNeura Team, 1999, 2000

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

#ifndef COMPAT_H
#define COMPAT_H

#include <string>
#include <iostream>

#ifdef __GNUC__
#if __GNUC__ < 3
        // check for stdlibc++v3 which does have ios_base
        #ifndef _CPP_BITS_IOSBASE_H
        typedef ios ios_base; // not currently defined in GCC
        #endif
#endif
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1300)
/*
Maarten: added this code here because Mirkosoft has the
nasty habit of #define min and max in stdlib.h (and windows.h)
I'm trying to undo this horrible macro magic (microsoft yet macrohard)
here. Sure hope it works
Olivier: this has been removed in .NET :) One step more standard...
*/
#pragma warning(disable:4786)

#include <stdlib.h>

#ifdef min
#undef min
#undef max // as they come in std::pairs
#endif

// add min and max to std...
namespace std
{
    template <class T> const T& min(const T& a, const T& b)
    {
        if(a < b)
            return a;
        // else
        return b;
    }

    template <class T> const T& max(const T& a, const T& b)
    {
        if(a > b)
            return a;
        // else
        return b;
    }
}

#endif
        // _MSC_VER
#endif
