/*
 * Copyright (C) DOLPHIN Project-Team, Lille Nord-Europe, 2007-2008
 * (C) OPAC Team, LIFL, 2002-2008
 *
 * (c) Mostepha Redouane Khouadjia <mr.khouadjia@ed.univ-lille1.fr>, 2008
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

#ifndef EOEASYPSODVRP_H_
#define EOEASYPSODVRP_H_



#include <eo>

#include "eoPsoDVRPflight.h"

#include "eoPsoDVRPvelocity.h"

#include "eoPsoDVRPEvalFunc.h"

#include "eoPsoDVRPEncodeSwarm.h"

#include "eoDVRPSecondsElapsedContinue.h"

#include "eoPsoDVRPInit.h"

#include <eoPopEvalFunc.h>

#include "eoGlobal.h"





template<class POT>


class eoPeoEasyPsoDVRP:public eoPSO < POT >
{

public :


/*	eoPeoEasyPsoDVRP(eoPsoDVRPEncodeSwarm<POT> & _encoding,

			     // eoEvalFunc< POT > &_eval,
	                      eoPsoDVRPEvalFunc<POT> &_evalDvrp,
			      eoPsoDVRPvelocity < POT > &_velocity,
			      eoFlight  < POT > &_move,
			      eoContinue <POT> &_continuator,
			      eoParticleBestInit <POT> &_initBest
			      ):
			      encoding(_encoding),
				  eval (_eval),
			      loopEval(_eval),
			      popEval(loopEval),
			      velocity (_velocity),
			      move (_move),
			      continuator(_continuator),
			      initBest(_initBest)
			     {}
*/
eoPeoEasyPsoDVRP(eoPsoDVRPEncodeSwarm<POT> & _encoding,
				eoEvalFunc < POT > &_eval,
			    eoPopEvalFunc < POT > &_popEval,
                eoPsoDVRPvelocity < POT > &_velocity,
			    eoFlight  < POT > &_move,
			    eoContinue <POT> &_continuator,
			    eoGenContinue <POT> &_genContinuator,
			    eoParticleBestInit <POT> &_initBest
			     ):
			    encoding(_encoding),
			    eval (_eval),
			    loopEval(_eval),
			    popEval(loopEval),
			    velocity (_velocity),
 		        move (_move),
 		        continuator(_continuator),
 		        genContinuator(_genContinuator),
 		        initBest(_initBest)
 		        {}


void operator()(eoPop < POT> &_pop)
{


double _TIME_STEP = TIME_STEP;


try

{
	eoPop<POT> empty_pop;
do{



	encoding.initParticleToItsBest(_pop);

	if(_TIME_STEP < TIME_CUTOFF * TIME_DAY){

		if(_TIME_STEP==0)
			loadNewCustomers (TIME_DAY,TIME_CUTOFF * TIME_DAY);


		else
		    loadNewCustomers(_TIME_STEP, TIME_SLICE);


		PrintLastCustomers();

		encoding(_pop,_TIME_STEP,0);
		popEval(empty_pop, _pop);
		initBest.apply(_pop);
        velocity.getTopology().setup(_pop);

	  }

	genContinuator.totalGenerations(genContinuator.totalGenerations());

      do
	 {
	  velocity.apply(_pop);
      move.apply(_pop);
      popEval(empty_pop,_pop);
      velocity.updateNeighborhood(_pop);

     }
     while (genContinuator(_pop));


	encoding.commitOrders(_pop,_TIME_STEP,TIME_SLICE);
	velocity.getTopology().setup(_pop);

	_TIME_STEP += TIME_SLICE ;

	cout<<_TIME_STEP<< "     "<<endl;

}while (continuator(_pop));

	closeTours(_pop);
	velocity.getTopology().setup(_pop);


	//_TIME_STEP += TIME_SLICE ;
	//cout<<endl<<getNodeRank() << "   "<<_TIME_STEP<< "     "<<endl;

//}while(_TIME_STEP < TIME_DAY);
}
catch (std::exception & e)
{
  std::string s = e.what ();
  s.append (" in eoEasyPSO");
  throw std::runtime_error (s);

}


}


private:

	eoPsoDVRPEncodeSwarm <POT> & encoding;
	eoEvalFunc < POT > &eval;
	eoPopLoopEval<POT> loopEval;
	eoPopEvalFunc<POT>& popEval;
	eoPsoDVRPvelocity < POT > &velocity;
	eoFlight < POT > &move;
	eoContinue< POT > &continuator;
	eoGenContinue < POT > & genContinuator;
	eoParticleBestInit <POT> &initBest;

	class eoDummyEval : public eoEvalFunc<POT>
	 {
	  public:
	  void operator()(POT & _po)
	  {}
	 }dummyEval;


};



void print(eoPop <eoParticleDVRP> &pop,std::ostream &_os)
{

	 for (size_t i =0, size = pop.size(); i< size ; ++i)
	{
	_os <<endl<<'\t'<<"---------------------- The Particle ["<<i<<"]----------------------"<< endl;

	pop[i].printCurrentPosition(_os);

	 pop[i].printRoutesOn(_os);

	pop[i].printBestPosition(_os);

	pop[i].printBestRoutesOn(_os);
	}


};



void printRoutes(eoPop <eoParticleDVRP> &pop,std::ostream &_os)
{

	 for (size_t i =0, size = pop.size(); i< size ; ++i)
		{

		 _os<<endl<<'\t'<<"---------------------- The Particle ["<<i<<"]----------------------"<< endl;

		 _os<<endl<<"The current fitness of the particule : "<< pop[i].best()<<endl;

		 pop[i].printRoutesOn(_os);

		}



}



void printBestRoutes(eoPop <eoParticleDVRP> &pop, std::ostream &_os)
{

	 for (size_t i =0, size = pop.size(); i< size ; ++i)
		{

		_os <<endl<<'\t'<<"---------------------- The Particle ["<<i<<"]----------------------"<< endl;

		_os<<endl<<"The best fitness of the particule : "<< pop[i].best()<<endl;

		 pop[i].printBestRoutesOn(_os);

		}

}


void printBestParticle(eoDVRPStarTopology<eoParticleDVRP> &topology, unsigned _seed, std::ostream &_os)
{

	_os <<endl<<'\t'<<"---------------------- THE  BEST PARTICLE OF SWARM ----------------------"<< endl;

	_os<<endl<<"Seed  "<<_seed<<endl;

	topology.best().printCurrentPosition(_os);

	topology.best().printRoutesOn(_os);

	topology.best().printBestPosition(_os);

	topology.best().printBestRoutesOn(_os);

    topology.best().printfirstBestTimeService(_os);

    _os<<endl<<"********************************************************************************************"<<endl;


}


	 void printVelocities(eoPop <eoParticleDVRP> &pop, std::ostream &_os)
	 {
		 for (size_t i =0, size = pop.size(); i<size ; ++i)

		 {
			 _os<<endl<<'\t'<<"----------------------The Velocity of Particule ["<< i <<"]----------------------"<< endl;

			 pop[i].printVelocities(_os);

		 }
	 };



	 void closeTours(eoParticleDVRP & _po)
		  {
			  for(size_t i = 0 , size = _po.pRoutes.size(); i < size; ++i)

				   _po.pRoutes[i].push_back(0);


			  for(size_t i = 0 , size = _po.bestRoutes.size(); i < size; ++i)

			       _po.bestRoutes[i].push_back(0);



		  }


	 void closeTours(eoPop <eoParticleDVRP> & _pop)
	 		  {

	 			  for(size_t i = 0 , size = _pop.size(); i < size; ++i)

	 				  closeTours(_pop[i]);


	 		  }



	 void resetSwarmToBestParticule(eoPop<eoParticleDVRP>&  _pop, eoParticleDVRP _po )
	 	 	  {

	 		 	for(size_t i = 0 , size = _pop.size() ; i < size ; i++)

	 		 		_pop[i] = _po ;

	 	 	  }







		void HillClimbingSearch( eoParticleDVRP & _po, unsigned sizeNeighboor = 1 )
		   {

		    	Routes neighborhoodSolution ;

		    	double depotTimeWindow = clients[0].durationService;

		    	double dueTime ;

		     	bool validNeighborSolution ;

		     	neighborhoodSolution = _po.pRoutes ;

		    	validNeighborSolution =  false;

		     	 for( size_t tr = 0; tr < neighborhoodSolution.size(); tr++)

		     	   {

		     		if(_po.serviceIsProgressCurrentPosition(tr))


		     		{

		     		   unsigned lastServedCustomer = _po.IndexLastServedCustomerCurrentPosition(tr) ;

		     		   unsigned positionLastCustomer = _po.planifiedCustomers[lastServedCustomer].pRouting.routePosition -1;

		     		  swapCustomers(neighborhoodSolution[tr],positionLastCustomer);

		     		   dueTime = getTimeOfService(neighborhoodSolution[tr], _po.planifiedCustomers[lastServedCustomer].id,_po.planifiedCustomers[lastServedCustomer].pRouting.serviceTime)

		     		   			 + distance(dist, neighborhoodSolution[tr].back(),0);


		     	   }else

		     	   {
		     		  swapCustomers(neighborhoodSolution[tr], 0 );

		     		   dueTime = getTimeOfService(neighborhoodSolution[tr], 0 , _po.firstTimeServiceCurrentPosition[tr]) + distance(dist, neighborhoodSolution[tr].back(),0);
		     	   }


		     		  if (dueTime <= depotTimeWindow)

		     			{ validNeighborSolution = true ;

		     			  break ;
		     			}


		     	   }




		if(validNeighborSolution)
		{

			double neighboorFitness  = computTourLength(neighborhoodSolution);

		//	std::cout<< " _po.fitness()  " <<  _po.fitness() << "   neighboorFitness    "<< neighboorFitness << endl ;

			if( neighboorFitness  < _po.fitness() )

			{
				_po.pRoutes = neighborhoodSolution;

				_po.reDesign() ;

				_po.fitness(neighboorFitness);

			}


		}


	}


		void HillClimbingSearch(eoPop<eoParticleDVRP>&  _pop, unsigned sizeNeighboor = 1 )
		{


			for(size_t i = 0,  size = _pop.size();  i < size ; i ++ )


				HillClimbingSearch(_pop[i], sizeNeighboor) ;



		}

		bool TwoOptimalAlgorithm (eoParticleDVRP & _po)

				{

						unsigned lastServedCustomerPosition;

						bool twoOptFeasible, changehappen = false ;

						double dueTime, depotTimeWindow = clients[0].durationService;



						for (size_t tr = 0, size = _po.pRoutes.size() ; tr < size ; tr++)

							{

							   Route routeOpt = _po.pRoutes[tr];

							    twoOptFeasible   =false;


			    	     		if(_po.serviceIsProgressCurrentPosition(tr))
			    	     		{

			    	     			unsigned lastServedCustomer = _po.IndexLastServedCustomerCurrentPosition(tr) ;

			    	     			lastServedCustomerPosition =  _po.planifiedCustomers[lastServedCustomer].pRouting.routePosition -1;

			    	     			if (twoOptOnRoute(routeOpt,lastServedCustomerPosition))

			    	     				 {

			    	     				  dueTime = getTimeOfService(routeOpt, _po.planifiedCustomers[lastServedCustomer].id,_po.planifiedCustomers[lastServedCustomer].pRouting.serviceTime)

			    	     				 					+ distance(dist, routeOpt.back(),0);

			    	     				 twoOptFeasible = true;
			    	     				 }
			    	     		}


			    	     		else
			    	     		{
			    	     			lastServedCustomerPosition = 0;

			    	     				if (twoOptOnRoute(routeOpt,lastServedCustomerPosition))
			    	     				{

			    	     					dueTime = getTimeOfService(routeOpt, 0 , _po.firstTimeServiceCurrentPosition[tr]) + distance(dist, routeOpt.back(),0);

			    	     					twoOptFeasible = true;

			    	     				}

			    	     		}


			    	     		if (twoOptFeasible && dueTime <= depotTimeWindow)

			    	     			{_po.pRoutes[tr] = routeOpt;

			    	     			  changehappen = true;
			    	     			}


						}

						if (changehappen)

							{
							   _po.reDesign() ;

							   _po.invalidate();

							   return true;


							}

						return false;


					}



				void TwoOptimalAlgorithm (eoPop<eoParticleDVRP> & _pop)
				{


					for(size_t i = 0, size = _pop.size(); i < size ; i++)

						TwoOptimalAlgorithm(_pop[i]);


				}



				bool ThreeOptimalAlgorithm (eoParticleDVRP & _po)

						{

								unsigned lastServedCustomerPosition;

								bool threeOptFeasible, changehappen = false ;

								double dueTime, depotTimeWindow = clients[0].durationService;



								for (size_t tr = 0, size = _po.pRoutes.size() ; tr < size ; tr++)

									{

									   Route routeOpt = _po.pRoutes[tr];

									    threeOptFeasible   =false;


					    	     		if(_po.serviceIsProgressCurrentPosition(tr))
					    	     		{

					    	     			unsigned lastServedCustomer = _po.IndexLastServedCustomerCurrentPosition(tr) ;

					    	     			lastServedCustomerPosition =  _po.planifiedCustomers[lastServedCustomer].pRouting.routePosition -1;

					    	     			if (threeOptOnRoute(routeOpt,lastServedCustomerPosition))

					    	     				 {

					    	     				  dueTime = getTimeOfService(routeOpt, _po.planifiedCustomers[lastServedCustomer].id,_po.planifiedCustomers[lastServedCustomer].pRouting.serviceTime)

					    	     				 					+ distance(dist, routeOpt.back(),0);

					    	     				 threeOptFeasible = true;
					    	     				 }
					    	     		}


					    	     		else
					    	     		{
					    	     			lastServedCustomerPosition = 0;

					    	     				if (threeOptOnRoute(routeOpt,lastServedCustomerPosition))
					    	     				{

					    	     					dueTime = getTimeOfService(routeOpt, 0 , _po.firstTimeServiceCurrentPosition[tr]) + distance(dist, routeOpt.back(),0);

					    	     					threeOptFeasible = true;

					    	     				}

					    	     		}


					    	     		if (threeOptFeasible && dueTime <= depotTimeWindow)

					    	     			{_po.pRoutes[tr] = routeOpt;

					    	     			  changehappen = true;
					    	     			}


								}

								if (changehappen)

									{
									   _po.reDesign() ;

									   _po.invalidate();

									   return true;


									}

								return false;


							}


				void ThreeOptimalAlgorithm (eoPop<eoParticleDVRP> & _pop)
				{


					for(size_t i = 0, size = _pop.size(); i < size ; i++)

						ThreeOptimalAlgorithm(_pop[i]);


				}


		void VariableNeighborhoodSearch(eoParticleDVRP & _po, unsigned neighborSize)
		{
			Routes neighborhoodSolution ;

			double  dueTime, depotTimeWindow = clients[0].durationService;

			bool validNeighborSolution ;

			neighborhoodSolution = _po.pRoutes ;

			validNeighborSolution =  false;

			unsigned neighborhoodIndex = 1;

		for( size_t tr = 0; tr < neighborhoodSolution.size(); tr++)

		 {

			if(_po.serviceIsProgressCurrentPosition(tr))

	     		{
				 unsigned lastServedCustomer = _po.IndexLastServedCustomerCurrentPosition(tr) ;

	        	 unsigned positionLastCustomer = _po.planifiedCustomers[lastServedCustomer].pRouting.routePosition -1;

	        	for(size_t k = 0 ; k <= neighborhoodIndex ; k++)

	        		swapCustomers(neighborhoodSolution[tr],positionLastCustomer);


	        		dueTime = getTimeOfService(neighborhoodSolution[tr], _po.planifiedCustomers[lastServedCustomer].id,_po.planifiedCustomers[lastServedCustomer].pRouting.serviceTime)

	    		   			   + distance(dist, neighborhoodSolution[tr].back(),0);


				 }else

				 {
					 for(size_t k = 0 ; k <= neighborhoodIndex ; k++)

	        		     swapCustomers(neighborhoodSolution[tr], 0);

					 dueTime = getTimeOfService(neighborhoodSolution[tr], 0 , _po.firstTimeServiceCurrentPosition[tr]) + distance(dist, neighborhoodSolution[tr].back(),0);
			     }


				 if (dueTime <= depotTimeWindow)

				    {

					 double neighboorFitness  = computTourLength(neighborhoodSolution);

			        if( neighboorFitness  < _po.fitness() )

				 		{
				    	   _po.pRoutes = neighborhoodSolution;

				    	   _po.reDesign() ;

				    	    _po.fitness(neighboorFitness);

				    	    if(TwoOptimalAlgorithm(_po))

				    	    {
				    	       _po.computCurrentTourLength();

				    	       _po.fitness(_po.toursLength());

				    	       neighborhoodIndex = 1;


				    	       continue ;
				    	   	}

				 		}

	    			}

				 neighborhoodIndex = (neighborhoodIndex + 1) % neighborSize ;
		 }

	 }

		void VariableNeighborhoodSearch(eoPop<eoParticleDVRP>& _pop, unsigned neighborSize)
		{


			for(size_t i = 0, size = _pop.size(); i < size ; i++)


				VariableNeighborhoodSearch(_pop[i], neighborSize);


		}


double avragePopFitness(eoPop<eoParticleDVRP>& _pop)
{
	double average = 0.0;

	for(size_t i = 0, size = _pop.size(); i < size ; i ++)
	{

		average += _pop[i].best();


	}

	return (average/_pop.size());


}

		bool AllCustomersAttributed(eoDVRPStarTopology<eoParticleDVRP> &topology)
		{
			size_t i = 0;

			while (i < topology.best().size())

			{
				if(!topology.best().planifiedCustomers[i].bestRouting.is_served)

					return false;

				i++;

			}

			return true;


		}

#endif /*EOEASYPSODVRP_H_*/
