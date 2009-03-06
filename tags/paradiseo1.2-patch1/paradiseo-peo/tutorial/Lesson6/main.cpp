/*
* <main.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, INRIA, 2008
*
* Clive Canape
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

#include <peo>

#include "route.h"
#include "route_init.h"
#include "route_eval.h"
#include "order_xover.h"
#include "city_swap.h"
#include "param.h"



#define POP_SIZE 10
#define NUM_GEN 100
#define CROSS_RATE 1.0
#define MUT_RATE 0.01


struct CoSearch
  {

    CoSearch(
      eoPop< Route >& A, eoPop< Route >& B,
      peoAsyncDataTransfer& asyncTransferA, peoAsyncDataTransfer& asyncTransferB
    )
        : transferA( A ), transferB( B ),
        asyncDataTransferA( asyncTransferA ), asyncDataTransferB( asyncTransferB )
    {
    }

    void operator()()
    {

      for ( unsigned int index = 0; index < 100; index++ )
        {

          asyncDataTransferA();
          asyncDataTransferB();

          eoPop< Route > intermed;
          intermed = transferA;
          transferA = transferB;
          transferB = intermed;
        }
    }

    eoPop< Route >& transferA;
    eoPop< Route >& transferB;

    peoAsyncDataTransfer& asyncDataTransferA;
    peoAsyncDataTransfer& asyncDataTransferB;
  };


struct PushBackAggregation
  {

    void operator()( eoPop< Route >& A, eoPop< Route >& B )
    {

      for ( unsigned int index = 0; index < B.size(); index++ )
        {

          A.push_back( B[ index ] );
        }
    }
  };

int main( int __argc, char** __argv )
{

  peo :: init( __argc, __argv );

  loadParameters( __argc, __argv );

  RouteInit route_init;
  RouteEval full_eval;
  OrderXover crossover;
  CitySwap mutation;
  eoPop< Route > population( POP_SIZE, route_init );
  eoGenContinue< Route > eaCont( NUM_GEN );
  eoCheckPoint< Route > eaCheckpointContinue( eaCont );
  eoRankingSelect< Route > selectionStrategy;
  eoSelectNumber< Route > eaSelect( selectionStrategy, POP_SIZE );
  eoSGATransform< Route > transformA( crossover, CROSS_RATE, mutation, MUT_RATE );
  eoPlusReplacement< Route > eaReplace;
  RingTopology ring;
  eoPlusReplacement< Route > transferReplace;
  peoAsyncDataTransfer asyncEAEndPoint( population, population, ring, transferReplace );
  eaCheckpointContinue.add( asyncEAEndPoint );
  eoEasyEA< Route > eaAlg( eaCheckpointContinue, full_eval, eaSelect, transformA, eaReplace );
  peoWrapper paraEAAlg( eaAlg, population );
  asyncEAEndPoint.setOwner( paraEAAlg );

  eoPop< Route > populationB( POP_SIZE, route_init );
  eoGenContinue< Route > eaContB( NUM_GEN );
  eoCheckPoint< Route > eaCheckpointContinueB( eaContB );
  eoRankingSelect< Route > selectionStrategyB;
  eoSelectNumber< Route > eaSelectB( selectionStrategyB, POP_SIZE );
  RingTopology ringB;
  eoPlusReplacement< Route > transferReplaceB;
  peoAsyncDataTransfer asyncEAEndPointB( populationB, populationB, ringB, transferReplaceB );
  eaCheckpointContinueB.add( asyncEAEndPointB );
  eoSGATransform< Route > transformB	( crossover, CROSS_RATE, mutation, MUT_RATE );
  eoEasyEA< Route > eaAlgB( eaCheckpointContinueB, full_eval, eaSelectB, transformB, eaReplace );
  peoWrapper paraEAAlgB( eaAlgB, populationB );
  asyncEAEndPointB.setOwner( paraEAAlgB );


  eoPop< Route > A, B;
  PushBackAggregation pushBackA, pushBackB;

  peoAsyncDataTransfer coSearchEndPointA( A, A, ring, pushBackA );
  peoAsyncDataTransfer coSearchEndPointB( B, B, ringB, pushBackB );

  CoSearch coSearch( A, B, coSearchEndPointA, coSearchEndPointB );
  peoWrapper paraCoSearch( coSearch );
  coSearchEndPointA.setOwner( paraCoSearch );
  coSearchEndPointB.setOwner( paraCoSearch );



  peo::run();
  peo::finalize();

  return 0;
}
