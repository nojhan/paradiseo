// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoCyclicGenContinue.h"

// (c) OPAC Team, LIFL, 2002

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
   
   Cont	act: cahon@lifl.fr
*/

#ifndef eoCyclicGenContinue_h
#define eoCyclicGenContinue_h

#include <eoContinue.h>
#include <eoPop.h>

/** 
    This continuator returns false in a periodic way during generations. 
*/

template <class EOT> class eoCyclicGenContinue: public eoContinue <EOT> {

public:
  
  /**
     Constructor. The frequency is given in parameter.
   */

  eoCyclicGenContinue (unsigned _freq,
		       unsigned init_count = 0) :
    freq (_freq),
    count (init_count) {
  }

  /** 
      Return true only if the current number of performed generations
      modulo the frequency equals none.
   */

  virtual bool operator () (const eoPop <EOT> & pop) {
    
    count ++ ;
    return (count % freq) != 0 ;
  }

private:
  
  unsigned count, freq ;
} ;

#endif

