// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFunctorStore.h
// (c) Maarten Keijzer 2000, GeNeura Team, 2000
/*
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
  Marc.Schoenauer@polytechnique.fr
  mak@dhi.dk
*/
//-----------------------------------------------------------------------------

#ifndef _eoFunctorStore_h
#define _eoFunctorStore_h

#include <vector>
#include<algorithm>

#include "utils/eoLogger.h"

class eoFunctorBase;

/**
   eoFunctorStore is a class that stores functors that are allocated on the
   heap. This class can be used in factories to store allocated memory for
   dynamically created functors.

   @ingroup Utilities
*/
class eoFunctorStore
{
public:

    /// Default Ctor
    eoFunctorStore() {}

    // virtual destructor so we don't need to define it in derived classes
    virtual ~eoFunctorStore();

    /// Add an eoFunctorBase to the store
    template <class Functor>
    Functor& storeFunctor(Functor* r)
        {
#ifndef NDEBUG
        unsigned int existing = std::count( vec.begin(), vec.end(), r );
        if( existing > 0 ) {
             eo::log << eo::warnings << "WARNING: you asked eoFunctorStore to store the functor " << r << " " 
                     << existing + 1 << " times, a segmentation fault may occur in the destructor." << std::endl;
        }
#endif
            // If the compiler complains about the following line,
            // check if you really are giving it a pointer to an
            // eoFunctorBase derived object
            vec.push_back(r);
            return *r;
        }

private :

    /** no copying allowed */
    eoFunctorStore(const eoFunctorStore&);

    /** no assignment allowed */
    eoFunctorStore operator=(const eoFunctorStore&);

protected:
    std::vector<eoFunctorBase*> vec;
};

#endif
