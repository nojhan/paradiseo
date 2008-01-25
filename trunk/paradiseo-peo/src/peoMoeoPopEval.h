/*
* <peoMoeoPopEval.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#ifndef __peoMoeoPopEval_h
#define __peoMoeoPopEval_h

#include <queue>
#include <eoEvalFunc.h>

#include "core/messaging.h"
#include "core/peo_debug.h"
#include "peoAggEvalFunc.h"
#include "peoNoAggEvalFunc.h"


//! Parallel evaluation functor wrapper.

//! The peoMoeoPopEval represents a wrapper for creating a functor capable of applying in parallel
//! an EO-derived evaluation functor. The class offers the possibility of chosing between a single-function evaluation
//! and an aggregate evaluation function, including several sub-evalution functions.
template< class EOT > class peoMoeoPopEval : public Service, public eoPopEvalFunc<EOT>
  {

  public:

    //! Constructor function - an EO-derived evaluation functor has to be specified; an internal reference
    //! is set towards the specified evaluation functor.
    //!
    //! @param eoEvalFunc< EOT >& __eval_func - EO-derived evaluation functor to be applied in parallel on each individual of a specified population
    peoMoeoPopEval( eoEvalFunc< EOT >& __eval_func );

    //! Constructor function - a vector of EO-derived evaluation functors has to be specified as well as an aggregation function.
    //!
    //! @param const std :: vector< eoEvalFunc < EOT >* >& __funcs - vector of EO-derived partial evaluation functors;
    //! @param peoAggEvalFunc< EOT >& __merge_eval - aggregation functor for creating a fitness value out of the partial fitness values.
    peoMoeoPopEval( const std :: vector< eoEvalFunc < EOT >* >& __funcs, peoAggEvalFunc< EOT >& __merge_eval );

    //! Operator for applying the evaluation functor (direct or aggregate) for each individual of the specified population.
    //!
    //! @param eoPop< EOT >& __pop - population to be evaluated by applying the evaluation functor specified in the constructor.
    void operator()(eoPop< EOT >& __pop);
    void operator()( eoPop< EOT >& __dummy, eoPop< EOT >& __pop );

    //! Auxiliary function for transferring data between the process requesting an evaluation operation and the process that
    //! performs the actual evaluation phase. There is no need to explicitly call the function.
    void packData();

    //! Auxiliary function for transferring data between the process requesting an evaluation operation and the process that
    //! performs the actual evaluation phase. There is no need to explicitly call the function.
    void unpackData();

    //! Auxiliary function - it calls the specified evaluation functor(s). There is no need to explicitly call the function.
    void execute();

    //! Auxiliary function for transferring data between the process requesting an evaluation operation and the process that
    //! performs the actual evaluation phase. There is no need to explicitly call the function.
    void packResult();

    //! Auxiliary function for transferring data between the process requesting an evaluation operation and the process that
    //! performs the actual evaluation phase. There is no need to explicitly call the function.
    void unpackResult();

    //! Auxiliary function for notifications between the process requesting an evaluation operation and the processes that
    //! performs the actual evaluation phase. There is no need to explicitly call the function.
    void notifySendingData();

    //! Auxiliary function for notifications between the process requesting an evaluation operation and the processes that
    //! performs the actual evaluation phase. There is no need to explicitly call the function.
    void notifySendingAllResourceRequests();

  private:


    const std :: vector< eoEvalFunc < EOT >* >& funcs;
    std :: vector< eoEvalFunc < EOT >* > one_func;

    peoAggEvalFunc< EOT >& merge_eval;
    peoNoAggEvalFunc< EOT > no_merge_eval;

    std :: queue< EOT* >tasks;

    std :: map< EOT*, std :: pair< unsigned, unsigned > > progression;

    unsigned num_func;

    EOT sol;

    EOT *ad_sol;

    unsigned total;
  };


template< class EOT > peoMoeoPopEval< EOT > :: peoMoeoPopEval( eoEvalFunc< EOT >& __eval_func ) :

    funcs( one_func ), merge_eval( no_merge_eval )
{

  one_func.push_back( &__eval_func );
}


template< class EOT > peoMoeoPopEval< EOT > :: peoMoeoPopEval(

  const std :: vector< eoEvalFunc< EOT >* >& __funcs,
  peoAggEvalFunc< EOT >& __merge_eval

) : funcs( __funcs ), merge_eval( __merge_eval )
{}

template< class EOT > void peoMoeoPopEval< EOT >::operator()(eoPop< EOT >& __dummy, eoPop< EOT >& __pop )
{
  this->operator()(__pop);
}

template< class EOT > void peoMoeoPopEval< EOT >::operator()(eoPop< EOT >& __pop )
{
  for ( unsigned i = 0; i < __pop.size(); i++ )
    {
      __pop[ i ].fitness(typename EOT :: Fitness() );
      progression[ &__pop[ i ] ].first = funcs.size() - 1;
      progression[ &__pop[ i ] ].second = funcs.size();
      for ( unsigned j = 0; j < funcs.size(); j++ )
        {
          /* Queuing the 'invalid' solution and its associated owner */
          tasks.push( &__pop[ i ] );
        }
    }
  total = funcs.size() * __pop.size();
  requestResourceRequest( funcs.size() * __pop.size() );
  stop();
}


template< class EOT > void peoMoeoPopEval< EOT > :: packData()
{
  //  printDebugMessage ("debut pakc data");
  pack( progression[ tasks.front() ].first-- );

  /* Packing the contents :-) of the solution */
  pack( *tasks.front() );

  /* Packing the addresses of both the solution and the owner */
  pack( tasks.front() );
  tasks.pop(  );
}


template< class EOT > void peoMoeoPopEval< EOT > :: unpackData()
{
  unpack( num_func );
  /* Unpacking the solution */
  unpack( sol );
  /* Unpacking the @ of that one */
  unpack( ad_sol );
}


template< class EOT > void peoMoeoPopEval< EOT > :: execute()
{

  /* Computing the fitness of the solution */
  funcs[ num_func ]->operator()( sol );
}


template< class EOT > void peoMoeoPopEval< EOT > :: packResult()
{
//	std::cout<<"\nD";
  /* Packing the fitness of the solution */
  /* typedef typename PO < F >::Fitness Fitness;
   MOEOObjectiveVector ObjectiveVector;*/
  std::vector < double > object;
  unsigned len;
  object=sol.objectiveVector();
  len=object.size();
  pack (len);
  for (unsigned i = 0 ; i < len; i ++)
    pack (object[i]);



//  pack( sol.fitness() );
  /* Packing the @ of the individual */
  pack( ad_sol );
//  std::cout<<"\nF";
}


template< class EOT > void peoMoeoPopEval< EOT > :: unpackResult()
{
//  typename EOT :: Fitness fit;

  /* Unpacking the computed fitness */
//  unpack( fit );
  unsigned len;
  std::vector < double > object;

  unpack(len);
  object.resize(len);
  for (unsigned i = 0 ; i < len; i ++)
    unpack (object[i]);
  /* Unpacking the @ of the associated individual */
  unpack( ad_sol );


  /* Associating the fitness the local solution */
//  merge_eval( *ad_sol, object );
  ad_sol->objectiveVector(object);
  progression[ ad_sol ].second--;

  /* Notifying the container of the termination of the evaluation */
  if ( !progression[ ad_sol ].second )
    {

      progression.erase( ad_sol );
    }

  total--;
  if ( !total )
    {

      getOwner()->setActive();
      resume();
    }

}


template< class EOT > void peoMoeoPopEval< EOT > :: notifySendingData()
{}


template< class EOT > void peoMoeoPopEval< EOT > :: notifySendingAllResourceRequests()
{
  getOwner()->setPassive();
}


#endif
