// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoDistPopEvalFunc.h"

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
   
   Contact: cahon@lifl.fr
*/

#ifndef eoDistPopEvalFunc_h
#define eoDistPopEvalFunc_h

#include <eoPopEvalFunc.h>
#include <paradisEO/comm/messages/to/eoEOSendMessTo.h>

template <class EOT> class eoDistPopEvalFunc : public eoPopEvalFunc <EOT> {
  
public :
  
  /**
     Constructor
   */ 

  eoDistPopEvalFunc (eoListener <EOT> & _listen,
		     string & _label,
		     eoEvalFunc <EOT> & _eval
		     ) :
    listen (_listen),
    label (_label),
    eval (_eval) {
    
  }

  void operator () (eoPop <EOT> & _parents, eoPop <EOT> & _offspring) {
        
    int num_eval = 0 ; // How many distributed evaluators ?
    int old_size = _offspring.size () ;
    
    do {
      listen.update () ;
      for (int i = 0 ; i < listen.size () ; i ++) {
	if (listen [i].label () == label) 
	  num_eval ++ ;
      }
      
      if (num_eval == 0) {
	cout << "No [" << label << "] available ..." << endl ;
	cout << "Waiting for a few seconds ..." << endl ;
	sleep (2) ;
      }
    } while (num_eval == 0) ;
    

    // Partitioning ...
    
    int j = 0, l = 0 ;
    
    bool t [listen.size ()] ;
    for (int i = 0 ; i < listen.size () ; i ++)
      t [i] = false ;

    for (int i = 0 ; i < num_eval ; i ++) {
      
      eoPop <EOT> pop ;
      for (int k = 0 ; k < old_size / num_eval ; k ++)
	pop.push_back (_offspring [j ++]) ;
      
      if (i < old_size % num_eval)
	pop.push_back (_offspring [j ++]) ;
      
      // Next evaluator ...
      while (listen [l].label () != label) {
	
	l ++ ;
      }
     
      eoEOSendMessTo <EOT> mess (pop) ;
      mess (listen [l]) ;
      t [l ++] = true ;
    }
    
    // On standby of the returns
    _offspring.clear () ;
    
    //  while (_offspring.size () != old_size) {
    
      //      listen.wait () ;
    
    for (int i = 0 ; i < listen.size () ; i ++)
      if (t [i]) {
	listen [i].wait () ;
	//while (! listen [i].empty ()) {
	
	eoPop <EOT> & pop = listen [i].front () ;
	
	for (int m = 0 ; m < pop.size () ; m ++)
	  _offspring.push_back (pop [m]) ;
	
	listen [i].pop () ;
	
	//}
	//}
      }
  }
  
private :
  
  eoListener <EOT> & listen ;
  string label ; // String identifier of evaluators 
  eoEvalFunc <EOT> & eval ;
} ; 

#endif
