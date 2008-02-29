/*
* <peoAsyncIslandMig.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2008
*
* Sebastien Cahon, Alexandru-Adrian Tantar, Clive Canape
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
* peoData to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#ifndef __peoAsyncIslandMig_h
#define __peoAsyncIslandMig_h


#include <queue>

#include <utils/eoUpdater.h>
#include <eoContinue.h>
#include <eoSelect.h>
#include <eoReplacement.h>
#include <eoPop.h>


#include "core/messaging.h"
#include "core/eoPop_mesg.h"
#include "core/eoVector_mesg.h"
#include "core/topology.h"
#include "core/thread.h"
#include "core/cooperative.h"
#include "core/peo_debug.h"


//! @class peoAsyncIslandMig
//! @brief Specific class for a asynchronous migration 
//! @see Cooperative eoUpdater
//! @version 2.0
//! @date january 2008
template< class TYPESELECT, class TYPEREPLACE > class peoAsyncIslandMig : public Cooperative, public eoUpdater
  {

  public:

    //! @brief Constructor
	//! @param continuator & __cont
	//! @param selector <TYPE> & __select 
	//! @param replacement <TYPE> & __replace
	//! @param Topology& __topology
	peoAsyncIslandMig(
      continuator & __cont,
      selector <TYPESELECT> & __select,
      replacement <TYPEREPLACE> & __replace,
      Topology& __topology
    );

	//! @brief operator
    void operator()();
    //! @brief Function realizing packages
    void pack();
    //! @brief Function reconstituting packages
    void unpack();
    //! @brief Function packSynchronizeReq
    void packSynchronizeReq();

  private:
	//! @brief Function which sends some emigrants
    void emigrate();
    //! @brief Function which receives some immigrants
    void immigrate();

  private:
  	//! @param continuator & cont
  	//! @param selector <TYPESELECT> & select
  	//! @param replacement <TYPEREPLACE> & replace
  	//! @param Topology& topology
  	//! @param std :: queue< TYPEREPLACE > imm
  	//! @param std :: queue< TYPESELECT > em
  	//! @param std :: queue< Cooperative* > coop_em
    continuator & cont;	
    selector <TYPESELECT> & select;	
    replacement <TYPEREPLACE> & replace;	
    Topology& topology;		
    std :: queue< TYPEREPLACE > imm;
    std :: queue< TYPESELECT > em;
    std :: queue< Cooperative* > coop_em;
  };


template< class TYPESELECT , class TYPEREPLACE> peoAsyncIslandMig< TYPESELECT, TYPEREPLACE > :: peoAsyncIslandMig(

  continuator & __cont,
  selector <TYPESELECT> & __select,
  replacement <TYPEREPLACE> & __replace,
  Topology& __topology

) : select( __select ), replace( __replace ), topology( __topology ), cont(__cont)
{

  __topology.add( *this );
}


template< class TYPESELECT , class TYPEREPLACE> void peoAsyncIslandMig< TYPESELECT, TYPEREPLACE > :: pack()
{
  lock ();
  ::pack( coop_em.front()->getKey() );
  ::pack(em.front());
  coop_em.pop();
  em.pop();
  unlock();
}


template< class  TYPESELECT, class TYPEREPLACE> void peoAsyncIslandMig< TYPESELECT , TYPEREPLACE> :: unpack()
{
  lock ();
  TYPEREPLACE mig;
  ::unpack(mig);
  imm.push( mig );
  unlock();
}

template< class TYPESELECT , class TYPEREPLACE> void peoAsyncIslandMig< TYPESELECT, TYPEREPLACE > :: packSynchronizeReq()
{}

template< class TYPESELECT , class TYPEREPLACE> void peoAsyncIslandMig< TYPESELECT , TYPEREPLACE> :: emigrate()
{
  std :: vector< Cooperative* >in, out;
  topology.setNeighbors( this, in, out );

  for ( unsigned i = 0; i < out.size(); i++ )
    {

      TYPESELECT mig;
      select(mig);
      em.push( mig );
      coop_em.push( out[i] );
      send( out[i] );
      printDebugMessage( "sending some emigrants." );
    }
}


template< class  TYPESELECT, class TYPEREPLACE> void peoAsyncIslandMig< TYPESELECT , TYPEREPLACE> :: immigrate()
{

  lock ();
  {

    while ( !imm.empty() )
      {
        replace(imm.front() );
        imm.pop();
        printDebugMessage( "receiving some immigrants." );
      }
  }
  unlock();
}


template< class TYPESELECT , class TYPEREPLACE> void peoAsyncIslandMig< TYPESELECT , TYPEREPLACE > :: operator()()
{

  if (! cont.check())
    {

      emigrate();	
      immigrate();	
    }
}


#endif
