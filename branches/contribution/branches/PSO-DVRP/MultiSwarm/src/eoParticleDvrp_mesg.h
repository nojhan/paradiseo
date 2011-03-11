#ifndef EOPARTICLEDVRP_MESG_H_
#define EOPARTICLEDVRP_MESG_H_

#include <core/eoVector_mesg.h>
#include <eoPsoDVRP.h>



void pack (const eoParticleDVRP & _v)
{
	cout<<endl<<"packing particle  " <<"  Node  "<< getNodeRank() << "  "  << _v.fitness() << "  " << _v.best() << endl;
	if (_v.invalid())
    {
      pack((unsigned)0);
   }
  else
    {
      pack((unsigned)1);
      pack (_v.fitness ());
      cout << _v.fitness() << endl;
      pack (_v.best());
      cout << _v.best() << endl;
    }

  unsigned len = _v.size();
  pack (len);
  cout<<"len  "<<len<<endl;
  for (unsigned i = 0 ; i < len; i ++)
  { pack (_v [i]);
  cout<<"_v[i]  "<<_v[i]<<endl;
  }

  for (unsigned i = 0 ; i < len; i ++)
    pack (_v.bestPositions[i]);
  for (unsigned i = 0 ; i < len; i ++)
    pack (_v.velocities[i]);

  for(unsigned i=0; i < len ; i++)

    {pack(_v.planifiedCustomers[i].id);
    pack(_v.planifiedCustomers[i].pRouting.route);
    pack(_v.planifiedCustomers[i].pRouting.routePosition);
    pack(_v.planifiedCustomers[i].pRouting.is_served);
    pack(_v.planifiedCustomers[i].pRouting.serviceTime);

    pack(_v.planifiedCustomers[i].bestRouting.route);
    pack(_v.planifiedCustomers[i].bestRouting.routePosition);
    pack(_v.planifiedCustomers[i].bestRouting.is_served);
    pack(_v.planifiedCustomers[i].bestRouting.serviceTime);

    pack(_v.planifiedCustomers[i].velocity);
    }

  unsigned pSizeTours,pSingleTour;
  pSizeTours= _v.pRoutes.size();
  pack(pSizeTours);
  for(unsigned i = 0; i < pSizeTours; i++)
  {
	  pSingleTour = _v.pRoutes[i].size();
	  pack(pSingleTour);
	  for(unsigned j = 0; j < pSingleTour; j++)
	    pack(_v.pRoutes[i][j]);

  }

    unsigned  bestSizeTours, bestSingleTour;
    bestSizeTours = _v.bestRoutes.size();
    pack(bestSizeTours);
    for(unsigned i = 0; i < bestSizeTours; i++)
    {
  	  bestSingleTour = _v.bestRoutes[i].size();
  	  pack(bestSingleTour);
  	  for(unsigned j = 0; j < bestSingleTour; j++)
  		  pack(_v.bestRoutes[i][j]);
    }



  pack(_v.pLength);
  pack(_v.bestLength);

  for(unsigned i=0; i < len ; i++)
   pack(_v.firstTimeServiceCurrentPosition[i]);

  for(unsigned i=0; i < len ; i++)
   pack(_v.firstTimeServiceBestPosition[i]);

}








void unpack (eoParticleDVRP & _v)
{


  unsigned valid;
  unpack(valid);

  if (! valid)
    {
      _v.invalidate();
      _v.invalidateBest();

    }
  else
    {
      double fit;
      unpack (fit);
      cout<<"fit  "<<fit<<endl;
      _v.fitness (fit);
      unpack(fit);
      cout<<"fit  "<<fit<<endl;
      _v.best(fit);

    }
  unsigned len;
  unpack (len);

  _v.resize (len);

  cout<<"len  "<<len<<endl;

  for (unsigned i = 0 ; i < len; i ++)
    {unpack (_v [i]);

    cout<<"_v[i]  "<<_v[i]<<endl;
    }

  _v.bestPositions.resize(len);
  for (unsigned i = 0 ; i < len; i ++)
    unpack (_v.bestPositions[i]);

  _v.velocities.resize (len);
  for (unsigned i = 0 ; i < len; i ++)
    unpack (_v.velocities[i]);

  _v.planifiedCustomers.resize(len);

  for(unsigned i=0; i < len ; i++)

     {
	  unpack(_v.planifiedCustomers[i].id);
	  cout<<"_v[i]  "<<_v.planifiedCustomers[i].id<<endl;
      unpack(_v.planifiedCustomers[i].pRouting.route);
      unpack(_v.planifiedCustomers[i].pRouting.routePosition);
      unpack(_v.planifiedCustomers[i].pRouting.is_served);
      unpack(_v.planifiedCustomers[i].pRouting.serviceTime);

      unpack(_v.planifiedCustomers[i].bestRouting.route);
      unpack(_v.planifiedCustomers[i].bestRouting.routePosition);
      unpack(_v.planifiedCustomers[i].bestRouting.is_served);
      unpack(_v.planifiedCustomers[i].bestRouting.serviceTime);

      unpack(_v.planifiedCustomers[i].velocity);



     }



  unsigned pSizeTours, pSingleTour;

  unpack(pSizeTours);
  _v.pRoutes.resize(pSizeTours);

  for(unsigned i = 0; i < pSizeTours; i++)
   {
 	  unpack(pSingleTour);
   	 _v.pRoutes[i].resize(pSingleTour);

 	  for(unsigned j = 0; j < pSingleTour; j++)
 	  { unpack(_v.pRoutes[i][j]);

 	    cout<<_v.pRoutes[i][j]<< "  ";
 	  }

 	  cout<<endl;

   }

    unsigned bestSizeTours,bestSingleTour;
    unpack(bestSizeTours);
    _v.bestRoutes.resize(bestSizeTours);

     for(unsigned i = 0; i < bestSizeTours; i++)
     {
   	  unpack(bestSingleTour);
   	  _v.bestRoutes[i].resize(bestSingleTour);

   	  for(unsigned j = 0; j < bestSingleTour; j++)
   		  unpack(_v.bestRoutes[i][j]);
     }

  unpack(_v.pLength);
  unpack(_v.bestLength);

  _v.firstTimeServiceCurrentPosition.resize(len);
  for(unsigned i=0; i < len ; i++)
   unpack(_v.firstTimeServiceCurrentPosition[i]);

  _v.firstTimeServiceBestPosition.resize(len);

  for(unsigned i=0; i < len ; i++)
   unpack(_v.firstTimeServiceBestPosition[i]);

}





#endif /*EOPARTICLEDVRP_MESG_H_*/
