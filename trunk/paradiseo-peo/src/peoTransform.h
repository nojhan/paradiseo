/*
* <peoTransform.h>
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

#ifndef __peoTransform_h
#define __peoTransform_h

#include "core/thread.h"
#include "core/messaging.h"
#include "core/peo_debug.h"
#include "core/service.h"


extern int getNodeRank();


template< class EOT > class peoTransform : public Service, public eoTransform< EOT >
{

public:

  peoTransform(

    eoQuadOp< EOT >& __cross,
    double __cross_rate,
    eoMonOp< EOT >& __mut,
    double __mut_rate
  );

  void operator()( eoPop< EOT >& __pop );

  void packData();

  void unpackData();

  void execute();

  void packResult();

  void unpackResult();

  void notifySendingData();
  void notifySendingAllResourceRequests();

private:

  eoQuadOp< EOT >& cross;
  double cross_rate;

  eoMonOp< EOT >& mut;
  double mut_rate;

  unsigned idx;

  eoPop< EOT >* pop;

  EOT father, mother;

  unsigned num_term;
};

template< class EOT > peoTransform< EOT > :: peoTransform(

  eoQuadOp< EOT >& __cross,
  double __cross_rate,
  eoMonOp < EOT >& __mut,
  double __mut_rate

) : cross( __cross ), cross_rate( __cross_rate ), mut( __mut ), mut_rate( __mut_rate )
{}


template< class EOT > void peoTransform< EOT > :: packData()
{

  pack( idx );
  pack( pop->operator[]( idx++ ) );
  pack( pop->operator[]( idx++ ) );
}


template< class EOT > void peoTransform< EOT > :: unpackData()
{

  unpack( idx );
  unpack( father );
  unpack( mother );
}


template< class EOT > void peoTransform< EOT > :: execute()
{

  if ( rng.uniform() < cross_rate ) cross( mother, father );

  if ( rng.uniform() < mut_rate ) mut( mother );
  if ( rng.uniform() < mut_rate ) mut( father );
}


template< class EOT > void peoTransform< EOT > :: packResult()
{

  pack( idx );
  pack( father );
  pack( mother );
}


template< class EOT > void peoTransform< EOT > :: unpackResult()
{

  unsigned sidx;

  unpack( sidx );
  unpack( pop->operator[]( sidx++ ) );
  unpack( pop->operator[]( sidx ) );
  num_term += 2;

  // Can be used with an odd size
  if ( num_term == 2*(pop->size()/2) )
  {

    getOwner()->setActive();
    resume();
  }
}


template< class EOT > void peoTransform< EOT > :: operator()( eoPop < EOT >& __pop )
{

  printDebugMessage( "peoTransform: performing the parallel transformation step." );
  pop = &__pop;
  idx = 0;
  num_term = 0;
  requestResourceRequest( __pop.size() / 2 );
  stop();
}


template< class EOT > void peoTransform< EOT > :: notifySendingData()
{}


template< class EOT > void peoTransform< EOT > :: notifySendingAllResourceRequests()
{

  getOwner()->setPassive();
}


#endif
