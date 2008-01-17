// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// peoPop.h 
// (c) GeNeura Team, 1998
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
 */
//-----------------------------------------------------------------------------

#ifndef _peoPop_H
#define _peoPop_H

#include <algorithm>
#include <iostream>
#include <iterator> // needed for GCC 3.2
#include <vector>

// EO includes
#include <eoOp.h> // for eoInit
#include <eoPersistent.h>
#include <eoInit.h>
#include <utils/rnd_generators.h>  // for shuffle method

#include <data.h>

//#include "core/eoPop_mesg.h"

#include "core/eoVector_mesg.h"
#include "core/messaging.h"


template<class EOT>
class peoPop: public data, public eoPop<EOT>
{
public:

  virtual void pack () 
  {
  	std::cout << "[peoPop][pack]" << std::endl;
  	
  	//::pack((*this));  
  	 ::pack ((unsigned) this->size ());
  for (unsigned i = 0; i < this->size (); i ++)
    ::pack ((*this)[i]);
  }

  virtual void unpack () 
  {
  	// ::unpack((*this));
  	 
  	  unsigned n;

  ::unpack (n);
  this->resize (n);
  for (unsigned i = 0; i < n; i ++)
   ::unpack ((*this)[i]);
  }

};

/*
template <class EOT> void pack (const peoPop <EOT> & __pop) {

  pack ((unsigned) __pop.size ());
  for (unsigned i = 0; i < __pop.size (); i ++)
    pack (__pop [i]);
}

template <class EOT> void unpack (peoPop <EOT> & __pop) {

  unsigned n;

  unpack (n);
  __pop.resize (n);
  for (unsigned i = 0; i < n; i ++)
    unpack (__pop [i]);
}
*/
#endif

