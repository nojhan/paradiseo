// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "peoParaPopEval.h"

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

#ifndef __peoParaPopEval_h
#define __peoParaPopEval_h

#include <queue>
#include <eoEvalFunc.h>

#include "mess.h"
#include "peo_debug.h"
#include "peoAggEvalFunc.h"

template <class EOT> class peoParaPopEval : public peoPopEval <EOT> {

public :

  peoParaPopEval (eoEvalFunc <EOT> & __eval_func);
  
  /** Constructor */
  peoParaPopEval (const std :: vector <eoEvalFunc <EOT> *> & __funcs,
		  eoAggEvalFunc <EOT> & __merge_eval
		  );
  
  void operator () (eoPop <EOT> & __pop);

  void packData ();

  void unpackData ();
  
  void execute ();
  
  void packResult ();

  void unpackResult ();
  
  void notifySendingData ();
  void notifySendingAllResourceRequests ();

private :

  const std :: vector <eoEvalFunc <EOT> *> & funcs;

  std :: vector <eoEvalFunc <EOT> *> one_func;
  
  peoAggEvalFunc <EOT> & merge_eval;
  
  peoNoAggEvalFunc <EOT> no_merge_eval ;

  std :: queue <EOT *> tasks;

  std :: map <EOT *, std :: pair <unsigned, unsigned> > progression; 

  unsigned num_func;  

  EOT sol;

  EOT * ad_sol; 
  
  unsigned total;
};


template <class EOT> 
peoParaPopEval <EOT> :: peoParaPopEval (eoEvalFunc <EOT> & __eval_func
				  ) : funcs (one_func),
				      merge_eval (no_merge_eval) {
  
  one_func.push_back (& __eval_func);
}

template <class EOT> 
peoParaPopEval <EOT> :: eoParaPopEval (const std :: vector <eoEvalFunc <EOT> *> & __funcs,
				 eoAggEvalFunc <EOT> & __merge_eval
				 ) : funcs (__funcs),
				     merge_eval (__merge_eval) {

}


template <class EOT> 
void peoParaPopEval <EOT> :: operator () (eoPop <EOT> & __pop) {

  
  for (unsigned i = 0; i < __pop.size (); i ++) {
    
    /* ??? */
    __pop [i].fitness (typename EOT :: Fitness ());
    /* */
    progression [& __pop [i]].first = funcs.size () - 1;
    progression [& __pop [i]].second = funcs.size ();
    
    for (unsigned i = 0; i < funcs.size (); i ++)
    /* Queuing the 'invalid' solution and its associated owner */
    tasks.push (& __pop [i]);  
  }  

  total = funcs.size () * __pop.size ();
  requestResourceRequest (funcs.size () * __pop.size ());
  stop ();
}


template <class EOT> 
void peoParaPopEval <EOT> :: packData () {

  /* Ugly !*/
  pack (progression [tasks.front ()].first --);
  /* Packing the contents :-) of the solution */
  pack (* tasks.front ()); 
  /* Packing the addresses of both the solution and the owner*/
  pack (tasks.front ()); 
  tasks.pop ();
}

template <class EOT> 
void peoParaPopEval <EOT> :: unpackData () {

  /* Ugly ! */
  unpack (num_func);
  /* Unpacking the solution */
  unpack (sol);
  /* Unpacking the @ of that one */ 
  unpack (ad_sol);
}

template <class EOT> 
void peoParaPopEval <EOT> :: execute () {

  /* Computing the fitness of the solution */
  funcs [num_func] -> operator () (sol);
}

template <class EOT> 
void peoParaPopEval <EOT> :: packResult () {

  /* Packing the fitness of the solution */
  pack (sol.fitness ());
  /* Packing the @ of the individual */
  pack (ad_sol);
}

template <class EOT> 
void peoParaPopEval <EOT> :: unpackResult () {

  typename EOT :: Fitness fit;

  /* Unpacking the computed fitness */
  unpack (fit);
  /* Unpacking the @ of the associated individual */  
  unpack (ad_sol);
  /* Associating the fitness the local solution */
  merge_eval (* ad_sol, fit);

  progression [ad_sol].second --; 
  /* Notifying the container of the termination of the evaluation */  
  if (! progression [ad_sol].second)
    progression.erase (ad_sol);    
  
  total --;
  if (! total) {
    getOwner () -> setActive ();
    resume ();  
  }
}


template <class EOT>
void peoParaPopEval <EOT> :: notifySendingData () {

  
}

template <class EOT>
void peoParaPopEval <EOT> :: notifySendingAllResourceRequests () {

  getOwner () -> setPassive ();
}

#endif
