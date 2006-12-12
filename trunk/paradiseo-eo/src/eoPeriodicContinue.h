// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoFreqContinue.h"

// (c) OPAC Team, LIFL, Feb. 2006

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
   
   Contact: cahon@lifl.fr  
*/

#ifndef __eoPeriodicContinue_h
#define __eoPeriodicContinue_h

#include <eoContinue.h>
#include <eoPop.h>

template <class EOT> class eoPeriodicContinue: public eoContinue <EOT> {

public:
  
  /** Constructor. The period is given in parameter. */  
  eoPeriodicContinue (unsigned __period, unsigned __init_counter = 0);

  /** It returns 'true' only if the current number of generations modulo
      the period doen't equal to zero. */  
  bool operator () (const eoPop <EOT> & pop);
    
private:

  unsigned period;
  
  unsigned counter;

};

template <class EOT> 
eoPeriodicContinue <EOT> :: eoPeriodicContinue (unsigned __period, unsigned __init_counter
						) : period (__period),
						    counter (__init_counter) {

}

template <class EOT> 
bool eoPeriodicContinue <EOT> :: operator () (const eoPop <EOT> & __pop) {
    
  return ((++ counter) % period) != 0 ;
}

#endif

