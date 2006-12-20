// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "peo_init.cpp"

// (c) OPAC Team, LIFL, August 2005

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <stdio.h>

#include "peo_init.h"
#include "peo_param.h"
#include "peo_debug.h"
#include "rmc.h"

namespace peo {

  int * argc;
  
  char * * * argv;

  void init (int & __argc, char * * & __argv) {

    argc = & __argc;
    
    argv = & __argv;
    
    /* Initializing the the Resource Management and Communication */
    initRMC (__argc, __argv);

    /* Loading the common parameters */ 
    loadParameters (__argc, __argv);
    
    /* */
    initDebugging ();
  }
}
