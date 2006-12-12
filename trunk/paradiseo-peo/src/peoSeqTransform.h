// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "peoSeqTransform.h"

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
   
   Contact: cahon@lifl.fr
*/

#ifndef __peoSeqTransform_h
#define __peoSeqTransform_h

#include "peoTransform.h"

template <class EOT> class peoSeqTransform : public peoTransform <EOT> {

public :

  /* Ctor */
  peoSeqTransform (eoTransform <EOT> & __trans);

  void operator () (eoPop <EOT> & __pop);

  virtual void packData () {}
  virtual void unpackData () {}

  virtual void execute () {}
  
  virtual void packResult () {}
  virtual void unpackResult () {}

private :
  
  eoTransform <EOT> & trans;

};

template <class EOT> 
peoSeqTransform <EOT> :: peoSeqTransform (eoTransform <EOT> & __trans
					  ) : trans (__trans) {

}

template <class EOT> 
void peoSeqTransform <EOT> :: operator () (eoPop <EOT> & __pop) {
  
  trans (__pop);
}

#endif
