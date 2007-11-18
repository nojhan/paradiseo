/*
* <peoSyncIslandMig.h>
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

#ifndef __peoSyncIslandMig_h
#define __peoSyncIslandMig_h


#include <queue>
#include <cassert>

#include <eoPeriodicContinue.h>

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


//! Class providing the basis for the synchronous island migration model.

//! The peoSyncIslandMig class offers the elementary basis for implementating a
//! synchronous island migration model - requires the specification of several basic
//! parameters, i.e. frequency of the migrations, selection and replacement strategies,
//! a topological model and the source and destination population for the migrating individuals.
//! The main difference as opposed to the asynchronous migration model is the synchronization step
//! performed after selecting and sending the emigrant individuals.
//!
//! The migration operator is called at the end of each generation of an evolutionary algorithms
//! as a checkpoint object - the following code exposes the structure of a classic evolutionary algorithm:
//!
//!	<table style="border:none; border-spacing:0px;text-align:left; vertical-align:top; font-size:8pt;" border="0">
//!	<tr><td><b>do</b> { &nbsp;</td> <td> &nbsp; </td></tr>
//!	<tr><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; select( population, offsprings ); &nbsp;</td> <td>// select the offsprings from the current population</td></tr>
//!	<tr><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; transform( offsprings ); &nbsp;</td> <td>// crossover and mutation operators are applied on the selected offsprings</td></tr>
//!	<tr><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; evaluate( offsprings ); &nbsp;</td> <td>// evaluation step of the resulting offspring</td></tr>
//!	<tr><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; replace( population, offsprings ); &nbsp;</td> <td>// replace the individuals in the current population whith individuals from the offspring population, according to a specified replacement strategy</td></tr>
//!	<tr><td>} <b>while</b> ( eaCheckpointContinue( population ) ); &nbsp;</td> <td>// checkpoint operators are applied on the current population, including the migration operator, if any specified </td></tr>
//!	</table>
//!
//! Constructing a synchronous island migration model requires having defined (1) a topological migration model,
//! (2) the control parameters of the migration process, (3) a checkpoint object associated with an evolutionary algorithm,
//! and (4) an owner object must be set. The owner object must be derived from the <b>Runner</b> class (for example
//! a peoEA object represents a possible owner).
//! A simple example is offered bellow:
//!
//!	<ol>
//!		<li> topological model to be followed when performing migrations: <br/>
//!		<br/>
//!		<table style="border:none; border-spacing:0px;text-align:left; vertical-align:top; font-size:8pt;" border="0">
//!		<tr><td>RingTopology migTopology; &nbsp;</td> <td>// a simple ring topological model - each island communicates with two other islands</td></tr>
//!		</table>
//!		</li>
//!
//!		<li> the continuation criterion, selection and replacement strategy etc. are defined: <br/>
//!		<br/>
//!		<table style="border:none; border-spacing:0px; font-size:8pt;" border="0">
//!		<tr><td>eoPop< EOT > population( POP_SIZE, popInitializer ); &nbsp;</td> <td>// population of individuals to be used for the evolutionary algorithm</td></tr>
//!		<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!		<tr><td>eoRandomSelect< EOT > migSelectStrategy; &nbsp;</td> <td>// selection strategy - in this case a random selection is applied</td></tr>
//!		<tr><td>eoSelectNumber< EOT > migSelect( migSelectStrategy, MIG_SIZE ); &nbsp;</td> <td>// number of individuals to be selected using the specified strategy</td></tr>
//!		<tr><td>eoPlusReplacement< EOT > migReplace; &nbsp;</td> <td>// immigration strategy - the worse individuals in the destination population are replaced by the immigrant individuals</td></tr>
//!		<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!		<tr><td>peoSyncIslandMig< EOT > syncMigration(
//!			<br/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MIG_FREQ, migSelect, migReplace, migTopology,
//!			<br/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; population, population
//!			<br/> ); &nbsp; </td>
//!			<td>// synchronous migration object - the emigrant individuals are selected from the same from population in which the immigrant individuals are being integrated </td></tr>
//!		</table>
//!		</li>
//!
//!		<li> creation of a checkpoint object as part of the definition of an evolutionary algoritm (details of th EA not given as being out of scope): <br/>
//!		<br/>
//!		<table style="border:none; border-spacing:0px;text-align:left; vertical-align:top; font-size:8pt;" border="0">
//!		<tr><td>... &nbsp;</td> <td> &nbsp; </td></tr>
//!		<tr><td>eoGenContinue< EOT > eaCont( NUM_GEN ); &nbsp;</td> <td>// the evolutionary algorithm will stop after NUM_GEN generations</td></tr>
//!		<tr><td>eoCheckPoint< EOT > eaCheckpointContinue( eaCont ); &nbsp;</td> <td>// number of individuals to be selected using the specified strategy</td></tr>
//!		<tr><td>... &nbsp;</td> <td> &nbsp; </td></tr>
//!		<tr><td>eaCheckpointContinue.add( syncMigration ); &nbsp;</td> <td>// adding the migration operator as checkpoint element</td></tr>
//!		<tr><td>... &nbsp;</td> <td> &nbsp; </td></tr>
//!		</table>
//!		</li>
//!
//!		<li> definition of an owner evolutionary algorithm (an object inheriting the <b>Runner</b> class): <br/>
//!		<br/>
//!		<table style="border:none; border-spacing:0px;text-align:left; vertical-align:top; font-size:8pt;" border="0">
//!		<tr><td>peoEA< EOT > eaAlg( eaCheckpointContinue, eaPopEval, eaSelect, eaTransform, eaReplace); &nbsp;</td> <td>// evolutionary algorithm having as checkpoint the eaCheckpointContinue object defined above </td></tr>
//!		<tr><td>syncMigration.setOwner( eaAlg ); &nbsp;</td> <td>// setting the evolutionary algorithm as owner of the migration object </td></tr>
//!		<tr><td>eaAlg( population ); &nbsp;</td> <td>// applying the evolutionary algorithm on a given population </td></tr>
//!		</table>
//!		</li>
//!	</ol>
//!
//! The source and the destination population for the migration object were specified as being the same, in step no. 2,
//! as we are usually interested in selecting the emigrants and integrating the immigrant individuals from and in, respectively, one unique
//! population, iteratively evolved by an evolutionary algorithm. There is no restriction in having two distinct populations
//! as source and destination for the emigrant and immigrant individuals respectively.
//!
//! The above steps only create a synchronous migration object associated to an evolutionary algorithm. The creation of several
//! islands requires the reiteration of the steps 2 through 4 for creating distinct algorithms, with distinct populations and
//! the associated distinctly parametrized migration objects. The interconnecting element is the underlying topology, defined at step 1
//! (the same C++ migTopology object has to be passed as parameter for all the migration objects, in order to interconnect them).
template< class EOT > class peoSyncIslandMig : public Cooperative, public eoUpdater
  {

  public:

    //! Constructor for the peoSyncIslandMig class; the characteristics of the migration model are defined
    //! through the specified parameters - out of the box objects provided in EO, etc., or custom, derived objects may be passed as parameters.
    //!
    //! @param unsigned __frequency - frequency of the migrations - the migrations occur periodically;
    //! @param eoSelect< EOT >& __select - selection strategy to be applied for constructing a list of emigrant individuals out of the source population;
    //! @param eoReplacement< EOT >& __replace - replacement strategy used for integrating the immigrant individuals in the destination population;
    //! @param Topology& __topology - topological model to be followed when performing migrations;
    //! @param eoPop< EOT >& __source - source population from which the emigrant individuals are selected;
    //! @param eoPop< EOT >& __destination - destination population in which the immigrant population are integrated.
    peoSyncIslandMig(
      unsigned __frequency,
      eoSelect< EOT >& __select,
      eoReplacement< EOT >& __replace,
      Topology& __topology,
      eoPop< EOT >& __source,
      eoPop< EOT >& __destination
    );

    //! Function operator to be called as checkpoint for performing the migration step. The emigrant individuals are selected
    //! from the source population and sent to the next island (defined by the topology object) while the immigrant
    //! individuals are integrated in the destination population. There is no need to explicitly call the function - the
    //! wrapper checkpoint object (please refer to the above example) will perform the call when required.
    void operator()();

    //! Auxiliary function dealing with sending the emigrant individuals. There is no need to explicitly call the function.
    void pack();
    //! Auxiliary function dealing with receiving immigrant individuals. There is no need to explicitly call the function.
    void unpack();

    //! Auxiliary function dealing with migration notifications. There is no need to explicitly call the function.
    void notifySending();


  private:

    void emigrate();
    void immigrate();


  private:

    eoPeriodicContinue< EOT > cont;
    eoSelect< EOT >& select; 	// selection strategy
    eoReplacement< EOT >& replace;	// replacement strategy
    Topology& topology; 		// neighboring topology

    // source and target populations
    eoPop< EOT >& source;
    eoPop< EOT >& destination;

    // immigrants & emigrants in the queue
    std :: queue< eoPop< EOT > > imm;
    std :: queue< eoPop< EOT > > em;

    std :: queue< Cooperative* > coop_em;

    sem_t sync;
  };


template< class EOT > peoSyncIslandMig< EOT > :: peoSyncIslandMig(

  unsigned __frequency,
  eoSelect< EOT >& __select,
  eoReplacement< EOT >& __replace,
  Topology& __topology,
  eoPop< EOT >& __source,
  eoPop< EOT >& __destination

) : cont( __frequency ), select( __select ), replace( __replace ), topology( __topology ), source( __source ), destination( __destination )
{

  __topology.add( *this );
  sem_init( &sync, 0, 0 );
}


template< class EOT > void peoSyncIslandMig< EOT > :: pack()
{

  lock ();

  pack( coop_em.front()->getKey() );
  pack( em.front() );
  coop_em.pop();
  em.pop();

  unlock();
}


template< class EOT > void peoSyncIslandMig< EOT > :: unpack()
{

  lock ();

  eoPop< EOT > mig;
  unpack( mig );
  imm.push( mig );

  unlock();

  sem_post( &sync );
}


template< class EOT > void peoSyncIslandMig< EOT > :: emigrate()
{

  std :: vector< Cooperative* > in, out;
  topology.setNeighbors( this, in, out );

  for ( unsigned i = 0; i < out.size(); i ++ )
    {

      eoPop< EOT > mig;
      select( source, mig );
      em.push( mig );
      coop_em.push( out[ i ] );
      send( out[ i ] );
      printDebugMessage( "sending some emigrants." );
    }
}


template< class EOT > void peoSyncIslandMig< EOT > :: immigrate()
{

  lock ();
  {

    assert( imm.size() );
    replace( destination, imm.front() ) ;
    imm.pop();
    printDebugMessage( "receiving some immigrants." );
  }
  unlock();
}


template< class EOT > void peoSyncIslandMig< EOT > :: operator()()
{

  if ( !cont( source ) )
    {

      // sending emigrants
      emigrate();
      stop();

      // synchronizing
      sem_wait( &sync );
      getOwner()->setActive();

      // receiving immigrants
      immigrate();
    }
}


template< class EOT > void peoSyncIslandMig< EOT > :: notifySending()
{

  lock ();
  {

    if ( imm.empty() )
      {

        printDebugMessage( "entering pasive mode\n" );
        getOwner()->setPassive();
      }
  }
  unlock();

  resume();
}


#endif
