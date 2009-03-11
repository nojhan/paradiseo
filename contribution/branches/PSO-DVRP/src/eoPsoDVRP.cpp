/*
 * Copyright (C) DOLPHIN Project-Team, INRIA Lille Nord-Europe, 2007-2008
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



#include "eoPsoDVRP.h"

#include "eoGlobal.h"



	 eoParticleDVRP::eoParticleDVRP(): eoVectorParticle <double,int,int> (), pLength(0.0), bestLength(0.0),pRoutes(),bestRoutes(),planifiedCustomers(),

	 									 firstTimeServiceCurrentPosition(), firstTimeServiceBestPosition(){}

	 eoParticleDVRP::~eoParticleDVRP (){}


	 void eoParticleDVRP::copy(const eoParticleDVRP & _po)
		 {
	         velocities = _po.velocities;

	         bestPositions = _po.bestPositions;

	         planifiedCustomers = _po.planifiedCustomers;

	         pRoutes  = _po.pRoutes;

	         bestRoutes = _po.bestRoutes;

	         pLength  = _po.pLength;

	         bestLength = _po.bestLength;

	         firstTimeServiceCurrentPosition = _po.firstTimeServiceCurrentPosition;

	         firstTimeServiceBestPosition = _po.firstTimeServiceBestPosition;

	         fitness(_po.pLength);

	         best (_po.bestLength);
		 }

	//	eoParticleDVRP::eoParticleDVRP(const eoParticleDVRP & _po){ operator= (_po);}


		eoParticleDVRP::eoParticleDVRP(const eoParticleDVRP & _po): eoVectorParticle <double,int,int> (_po)
		{
			copy(_po);
		}


	/*	eoParticleDVRP& eoParticleDVRP::operator= (const eoParticleDVRP& _po)
		{


			            eoVectorParticle<double,int,int>::operator = (_po); // inherit from the std::vector operator

		                velocities = _po.velocities;

		                bestPositions = _po.bestPositions;

		                planifiedCustomers = _po.planifiedCustomers;

			            pRoutes  = _po.pRoutes;

			            bestRoutes = _po.bestRoutes;

			            pLength  = _po.pLength;

			            bestLength = _po.bestLength;

			            firstTimeServiceCurrentPosition = _po.firstTimeServiceCurrentPosition;

			            firstTimeServiceBestPosition = _po.firstTimeServiceBestPosition;

			            fitness(_po.pLength);

			            best (_po.bestLength);



			        return *this;


			        //eoVectorParticle <double,int,int>::operator=(_po);
			        				//copy(_po);


			    }*/

		eoParticleDVRP& eoParticleDVRP::operator= (const eoParticleDVRP& _po)
		{
			eoVectorParticle <double,int,int>::operator=(_po);
			copy(_po);

		}

    std::string eoParticleDVRP::className () const {

        return "The class is eoParticleDVRP";

    }




     double eoParticleDVRP::bestToursLength () {

             return bestLength;

         }


         double eoParticleDVRP::toursLength () {

             return pLength;

         }


  void eoParticleDVRP::toursLength(const double _pLength)
         {

        	 pLength=_pLength;


         }

 void eoParticleDVRP::bestToursLength(const double _bestLength)
         {

	 		bestLength=_bestLength;


         }
         bool eoParticleDVRP::clean () {

             this->clear ();

             planifiedCustomers.clear();

             pRoutes.clear ();

             bestRoutes.clear();

             pLength = bestLength = 0.0;

             firstTimeServiceCurrentPosition.clear();

             firstTimeServiceBestPosition.clear();

             return true;

         }


         bool eoParticleDVRP::cleanCurrentRoutes () {


         	 pRoutes.clear ();

         	 firstTimeServiceCurrentPosition.clear();

         	 pLength = 0.0;


      	return true;


         }


         bool eoParticleDVRP::cleanBestRoutes (){


             bestRoutes.clear ();

             firstTimeServiceBestPosition.clear();

             bestLength =0.0;

             return true;

         }



         void eoParticleDVRP::printRoutesOn(std::ostream& _os) const

         {

        	  if (invalid())

        		  _os <<endl<<'\t'<<"CurrentParticleFitness :INVALID "<<endl;

        	  else

        		  _os <<endl<<'\t'<<"CurrentPositionFitness:"<< fitness() <<endl<<'\n'<<">>> The numbre of routes:"<<pRoutes.size()<<'\n'<<endl;

        	      printRoutes(pRoutes, _os);




         }



         void eoParticleDVRP::printBestRoutesOn(std::ostream& _os) const

         {

              	    if (invalidBest())

              	    	_os <<endl<<'\t'<<"BestPositionFitness :INVALID "<<endl;

              	     else

              	    	 _os <<endl<<'\t'<<"BestPositionFitness:"<< best()<<endl<<'\n'<<">>> The numbre of routes:"<<bestRoutes.size()<<'\n'<< endl;

              	         printRoutes(bestRoutes,_os);


           	}


         void eoParticleDVRP::printOn(std::ostream& _os) const
         {

        	 if(invalid())


        		 _os <<'\t'<<"ParticleFitness :INVALID ";

        	 _os << " The size :"<< size()<<endl;


        	 std::copy(begin(),end(), std::ostream_iterator<unsigned>(_os, " "));



         }


         void eoParticleDVRP::printBestOn(std::ostream& _os) const

         {
        	 if (invalidBest())


        		 _os <<endl<<'\t'<<"ParticleBestFitness :INVALID ";


        	 _os <<" The size :"<< bestPositions.size()<<endl;


        	 std::copy(bestPositions.begin(),bestPositions.end(), std::ostream_iterator<unsigned>(_os, " "));


          }


         void eoParticleDVRP::printVelocities (std::ostream& _os)
             {


        	 		_os <<endl<<"The velocities of Particle: "<<velocities.size()<<endl;


                	for ( size_t i = 0; i < velocities.size(); i++ )


                		_os<<'\t'<<planifiedCustomers[i].id <<'\t'<<velocities[i]<<'\t'<<planifiedCustomers[i].velocity<<endl;

             }

         void eoParticleDVRP::printfirstTimeService (std::ostream& _os)
        	 {

        	 	_os<<endl<<"The Leaving Time of vehicles in Current Position: "<<firstTimeServiceCurrentPosition.size()<<endl;


        	 	for (size_t i =0, size = firstTimeServiceCurrentPosition.size(); i < size; ++i)


        	 		_os<<firstTimeServiceCurrentPosition[i]<<'\t';

        		 _os<<endl;
        	 }

        void eoParticleDVRP::printfirstBestTimeService (std::ostream& _os)
         	 {

        	   _os <<endl<<"The Leaving Time of vehicles in Best Position: "<<firstTimeServiceBestPosition.size()<<endl;

         		 for (size_t i =0, size = firstTimeServiceBestPosition.size(); i < size; ++i)

        	 			 _os<<firstTimeServiceBestPosition[i]<<'\t';

        	 		_os<<endl;
        	 }



         void eoParticleDVRP::printCurrentPosition(std::ostream& _os)const
           {


        	_os <<endl<<"The Current position: "<<endl;

        	_os<<endl<<"ID_Customer   "<<"Tour   "<<"Position  "<<"is_served(Y/N)    "

        	<< "ServiceTime"<<endl;

         	for ( size_t i= 0; i < planifiedCustomers.size(); i++ )

             _os<<'\t'<<planifiedCustomers[i].id<<'\t'

             		  <<planifiedCustomers[i].pRouting.route<<'\t'

             		  <<planifiedCustomers[i].pRouting.routePosition<<'\t'

             		  <<clients[planifiedCustomers[i].id].availTime<< '\t'

             		  <<planifiedCustomers[i].pRouting.is_served<<'\t'<<'\t'

             		 <<planifiedCustomers[i].pRouting.serviceTime<<endl;


           }


         void eoParticleDVRP::printBestPosition(std::ostream& _os)const

                   {

                       _os <<endl<<"The Best position: "<<endl;

                      for ( size_t i= 0; i < planifiedCustomers.size(); i++ )

                    	  _os<<'\t'<<planifiedCustomers[i].id<<'\t'

                  	  <<planifiedCustomers[i].bestRouting.route<<'\t'

                    	  <<planifiedCustomers[i].bestRouting.routePosition<<'\t'

                    	  << clients[planifiedCustomers[i].id].availTime<<'\t'


                  	  <<planifiedCustomers[i].bestRouting.is_served<<'\t'<<'\t'

                    	  <<planifiedCustomers[i].bestRouting.serviceTime<<endl;


                   }



         void eoParticleDVRP:: setVelocities()
         {

        	 velocities.clear();

        	 for (size_t i =0, size = planifiedCustomers.size() ; i < size ; i++)

        	        velocities.push_back(planifiedCustomers[i].velocity);
         }




         void eoParticleDVRP::setCurrentPositions ()
         {

        	 this-> clear();

           	 for ( size_t i =0 ; i < planifiedCustomers.size() ; i++ )

        	     this->push_back(planifiedCustomers[i].pRouting.route);
         }


         void eoParticleDVRP::setBestPositions ()
                  {

                 	 bestPositions.clear();

                 	 for ( size_t i =0 ; i < planifiedCustomers.size()  ; i++ )

                 	      bestPositions.push_back(planifiedCustomers[i].bestRouting.route);
                  }



         void eoParticleDVRP::computToursLength()
              {

       			computCurrentTourLength();

        		computBestTourLength();


               }




         void eoParticleDVRP::computCurrentTourLength()

         {
           	 pLength = 0.0 ;

           	 for(size_t tr = 0, tsize = pRoutes.size(); tr < tsize; tr++ )

           		 {

           		   for (size_t i = 0, size = pRoutes[tr].size()-1; i < size; ++i )

           			     pLength+= distance(dist,pRoutes[tr][i],pRoutes[tr][i+1]);


           		 	pLength+=distance(dist,pRoutes[tr].back(),0);

           		 }

         }


         void eoParticleDVRP::computBestTourLength()

         {
             bestLength =0.0;


             for(size_t tr = 0, tsize = bestRoutes.size(); tr < tsize; ++tr )

                  {
                	for (size_t i = 0, size = bestRoutes[tr].size()-1; i < size; ++i )

                	bestLength+=distance(dist,bestRoutes[tr][i], bestRoutes[tr][i+1]);


                	bestLength+= distance(dist, bestRoutes[tr].back(),0);


                   }
         }



         void eoParticleDVRP::encodingCurrentPositionCheapestInsertion(double _tstep, double _tslice)  //Encoding with cheapest insertion algorithm
                  {


        	       Route newRoute;

        	       bool commitCustomer;

        	       particleRouting customer ;

        	       unsigned  cheapestTour, positionCheapestTour; // pas globale

           	       double cheapestTourCost, depotTimeWindow = clients[0].durationService;

        	      unsigned i = randTour(newCustomers.size());


        	       for (size_t k = 0, size = newCustomers.size(); k < size; k++)

        	       {

        	    	    customer.id = newCustomers[i];



        	            commitCustomer = false;

           	            for (unsigned tour = 0 ; tour < pRoutes.size() ; tour++ )

        	              {

        	            	 if(serviceIsProgressCurrentPosition(tour))

        	            	    {

        	            		 unsigned lastServedCustomer = IndexLastServedCustomerCurrentPosition(tour) ;

        	            		 unsigned positionLastCustomer = planifiedCustomers[lastServedCustomer].pRouting.routePosition -1 ;

        	            		 cheapestInsertionAlgorithm( pRoutes[tour], tour, planifiedCustomers[lastServedCustomer].id,

        	            				 					positionLastCustomer, planifiedCustomers[lastServedCustomer].pRouting.serviceTime,

        	            				 					customer.id, cheapestTour, cheapestTourCost,  positionCheapestTour,commitCustomer);


        	            	    }



        	            	 else


        	            		 cheapestInsertionAlgorithm (pRoutes[tour], tour, firstTimeServiceCurrentPosition[tour], customer.id,

        	            				 					cheapestTour, cheapestTourCost,  positionCheapestTour, commitCustomer);


        	              }


           	          double costNewInsertionDepot = distance(dist,0 , customer.id) +  distance (dist, customer.id,0) ;

           	           double  dueTimeDepot = _tstep + _tslice +  costNewInsertionDepot + clients[customer.id].durationService;


           	           if(commitCustomer && costNewInsertionDepot < cheapestTourCost  && dueTimeDepot <= clients[0].durationService && pRoutes.size() <= FLEET_VEHICLES)

           	               commitCustomer = false;





           	         if (!commitCustomer)

           	           {
           	        	 if (pRoutes.size() <= FLEET_VEHICLES)
           	                {
           	                	  Route newRoute = emptyRoute();

           	                 	  newRoute.push_back(customer.id);

           	                 	  pRoutes.push_back(newRoute);

           	                 	  firstTimeServiceCurrentPosition.push_back(_tstep +  _tslice);

           	                 	  customer.pRouting.route = pRoutes.size();

           	                 	  customer.pRouting.routePosition = pRoutes.back().size() ;

           	                }
           	                else

           	                 customer.pRouting.route =customer.pRouting.routePosition = -1;
           	           }
           	         else
           	         {

           	        	 pRoutes[cheapestTour].insert(pRoutes[cheapestTour].begin() + positionCheapestTour, customer.id );


           	        	 customer.pRouting.route = cheapestTour +1;


           	        	 customer.pRouting.routePosition = positionCheapestTour +1 ;



           	        	 for(size_t i = positionCheapestTour +1 , size = pRoutes[cheapestTour].size() ; i < size; i++)

           	        	   {

           	        	     for (size_t j = 0 , sizeP = this -> size() ; j < sizeP ; j++)


           	        	     {

           	        	    	 if(planifiedCustomers[j].id == pRoutes[cheapestTour][i])

           	        	    	 	{


           	        	    		 planifiedCustomers[j].pRouting.routePosition ++;


           	        	    		break ;


           	        	           	}


           	               	 }

  	           	         }
           	        }


       		 customer.velocity = randVelocity(pRoutes.size()-1);

          	 velocities.push_back(customer.velocity);

          	 this->push_back(customer.pRouting.route);

          	 customer.pRouting.is_served = false;

          	 customer.pRouting.serviceTime = -1;

          	 planifiedCustomers.push_back(customer);


          	 i = ( i + 1 ) % newCustomers.size();


       }


        invalidate ();

        invalidateBest();


     }

       //--------------------------------------------------------------------------------------------------
         //--------------------------------------------------------------------------------------------------

        void eoParticleDVRP::encodingCurrentPositionNearestInsertion(double _tstep, double _tslice)  //Encoding with nearest insertion algorithm
            {

              	       Route newRoute;

              	       bool commitCustomer;

              	       particleRouting customer ;

              	       unsigned  nearestTour, positionNearestTour; // pas globale

              	       double nearestTourCost, depotTimeWindow = clients[0].durationService;


              	       unsigned i = randTour(newCustomers.size());


             	       for (size_t k = 0, size = newCustomers.size(); k < size; k++)

            	          {
              	             customer.id = newCustomers[i];

              	             commitCustomer = false;

                 	         for (unsigned tour = 0 ; tour < pRoutes.size() ; tour++ )

              	              {

              	            	 if(serviceIsProgressCurrentPosition(tour))

              	            	    {

              	            		 unsigned lastServedCustomer = IndexLastServedCustomerCurrentPosition(tour) ;

              	            		 unsigned positionLastCustomer = planifiedCustomers[lastServedCustomer].pRouting.routePosition -1 ;

              	            		 nearestInsertionAlgorithm( pRoutes[tour], tour, planifiedCustomers[lastServedCustomer].id,

              	            			 					positionLastCustomer, planifiedCustomers[lastServedCustomer].pRouting.serviceTime,

              	            								customer.id, nearestTour, nearestTourCost,  positionNearestTour,commitCustomer);

              	            	    }

            	            	 else


              	            		 nearestInsertionAlgorithm (pRoutes[tour], tour, firstTimeServiceCurrentPosition[tour], customer.id,

              	            	    		 					nearestTour, nearestTourCost,  positionNearestTour, commitCustomer);

              	              }

                 	        double costNewInsertionDepot = distance(dist,0 , customer.id) +  distance (dist, customer.id,0) ;

                 	        double  dueTimeDepot = _tstep + _tslice +  costNewInsertionDepot;

               	           if( commitCustomer &&  costNewInsertionDepot < nearestTourCost  && dueTimeDepot <= clients[0].durationService && pRoutes.size() <= FLEET_VEHICLES)

               	        	      commitCustomer = false;



                 	         if (!commitCustomer)

                 	           {
                 	        	 if (pRoutes.size() <= FLEET_VEHICLES)
                 	                {
                 	                	  Route newRoute = emptyRoute();

                 	                 	  newRoute.push_back(customer.id);

                 	                 	  pRoutes.push_back(newRoute);

                 	                 	  firstTimeServiceCurrentPosition.push_back(_tstep + _tslice); //////////////

                 	                 	  customer.pRouting.route = pRoutes.size();

                 	                 	  customer.pRouting.routePosition = pRoutes.back().size() ;

                 	                }
                 	                else

                 	                 customer.pRouting.route =customer.pRouting.routePosition = -1;
                 	           }
                 	         else
                 	         {

                 	             pRoutes[nearestTour].insert(pRoutes[nearestTour].begin() + positionNearestTour, customer.id );

              	            	 customer.pRouting.route = nearestTour +1;

              	            	 customer.pRouting.routePosition = positionNearestTour +1 ;

              	            	 for(size_t i = positionNearestTour +1 , size = pRoutes[nearestTour].size() ; i < size; i++)

              	            	      {

              	          				   for (size_t j = 0 , sizeP = this -> size() ; j < sizeP ; j++)
              	            	           		{

              	          					   if(planifiedCustomers[j].id == pRoutes[nearestTour][i])

              	          					       {
              	          					    	   	planifiedCustomers[j].pRouting.routePosition ++;

              	            	           						   break ;

              	            	           		   }

              	            	   			   }
              	            	       }
                 	         }



             		 customer.velocity = randVelocity(pRoutes.size()-1);

                	 velocities.push_back(customer.velocity);

                	 this->push_back(customer.pRouting.route);

                	 customer.pRouting.is_served = false;

                	 customer.pRouting.serviceTime = -1;

                	 planifiedCustomers.push_back(customer);


                	 i = ( i + 1 ) % newCustomers.size();


             }


              invalidate ();

              invalidateBest();


           }



         void eoParticleDVRP::encodingCurrentPositionRandomInsertion(double _tstep, double _tslice)  // initialisation aléatoire
         {
        	 Route newRoute;

        	 bool commitCustomer;

        	 particleRouting customer ;

        	 unsigned positionNewCustomer ;

        	 double depotTimeWindow = clients[0].durationService;


        	unsigned i = randTour(newCustomers.size());


        	 for (size_t k = 0, size = newCustomers.size(); k < size; k++)

        		 {

        		   customer.id = newCustomers[i];

        		   commitCustomer = false;

        		   unsigned  tr = randTour(pRoutes.size());


        		   for (size_t j = 0 ; j < pRoutes.size() ; j++ )
        		   {


        			 double dueTime;


        			 if(serviceIsProgressCurrentPosition(tr))

        			  {
        				unsigned lastServedCustomer = IndexLastServedCustomerCurrentPosition(tr) ;

        				unsigned positionLastCustomer = planifiedCustomers[lastServedCustomer].pRouting.routePosition -1  ;

        				positionNewCustomer = randPosition(positionLastCustomer + 1 , pRoutes[tr].size());

          				dueTime  = getTimeOfService(pRoutes[tr],  planifiedCustomers[lastServedCustomer].id, planifiedCustomers[lastServedCustomer].pRouting.serviceTime, customer.id, positionNewCustomer)

         			               + distance (dist, pRoutes[tr].back(),0);


        			  }
        			   else


        			   {


        				   positionNewCustomer = randPosition(1, pRoutes[tr].size());


           				   dueTime  = getTimeOfService(pRoutes[tr],firstTimeServiceCurrentPosition[tr], customer.id, positionNewCustomer)

        				   			+ distance (dist, pRoutes[tr].back(),0);



        			   }




					  double demandTour =  getCapacityUsed(pRoutes[tr]) + clients[customer.id].demand;



        			   if (dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )

        			   {


        			   pRoutes[tr].insert(pRoutes[tr].begin() + positionNewCustomer, customer.id );

           			   customer.pRouting.route = tr+1;

           			   customer.pRouting.routePosition = positionNewCustomer +1 ;



           			   for(size_t i = positionNewCustomer +1 , size = pRoutes[tr].size() ; i < size; i++)

           			   {

           				   for (size_t j = 0 , sizeP = this -> size() ; j < sizeP ; j++)
           				   {

           					   if(planifiedCustomers[j].id == pRoutes[tr][i])
           					   {


           						   planifiedCustomers[j].pRouting.routePosition ++;

           						   break ;

           					   }

           				   }



           			   }


        		       commitCustomer = true ;

        		       break;


        			   }

        			   tr = ( tr + 1 ) % pRoutes.size();


        		   }

        		   if (!commitCustomer)

        			   if (pRoutes.size() < FLEET_VEHICLES)
        			   {

        				   Route newRoute = emptyRoute();

        		   	       newRoute.push_back(customer.id);

        		   	       pRoutes.push_back(newRoute);

        		   	       firstTimeServiceCurrentPosition.push_back(_tstep + _tslice);  /////////

           		   	       customer.pRouting.route = pRoutes.size();

           		   	       customer.pRouting.routePosition = pRoutes.back().size() ;

        			   }

        		   	   else

        		   	    	customer.pRouting.route =customer.pRouting.routePosition = -1;



        		 customer.velocity = randVelocity(pRoutes.size()-1);

        		 velocities.push_back(customer.velocity);

        		 this->push_back(customer.pRouting.route);

        		 customer.pRouting.is_served = false;

        		 customer.pRouting.serviceTime = -1;

           		 planifiedCustomers.push_back(customer);


                 i = ( i + 1 ) % newCustomers.size();

        		 }


        	 invalidate ();

        	 invalidateBest ();

            }



         void eoParticleDVRP::encodingCurrentPositionGeneralizedInsertion(double _tstep, double _tslice)
           {

         	 Route newRoute , GenRoute;

         	 bool commitCustomer;

      	     particleRouting customer ;

         	 unsigned generalizedTour, positionGeneralizedTour; // pas globale

         	 double generalizedTourCost;

         	 double depotTimeWindow = clients[0].durationService;

         	 unsigned i = randTour(newCustomers.size());

         	 for (size_t k = 0, size = newCustomers.size(); k < size; k++)

         	 {
         		 customer.id = newCustomers[i];

         		 GenRoute.clear();

         		 commitCustomer = false;



         		 for (unsigned tour = 0 ; tour < pRoutes.size() ; tour++ )

         		 {



         			 if(serviceIsProgressCurrentPosition(tour))

         			   {


         				 unsigned lastServedCustomer = IndexLastServedCustomerCurrentPosition(tour) ;


         				 unsigned positionLastCustomer = planifiedCustomers[lastServedCustomer].pRouting.routePosition -1 ;


         				 generalizedInsertionTypeOneAlgorithm( pRoutes[tour],GenRoute,tour,  planifiedCustomers[lastServedCustomer].id,


         						                      positionLastCustomer, planifiedCustomers[lastServedCustomer].pRouting.serviceTime,


         						                      customer.id, generalizedTour, positionGeneralizedTour, generalizedTourCost,  commitCustomer);



         				/* generalizedInsertionTypeOneAlgorithm( pRoutes[tour], tour, planifiedCustomers[lastServedCustomer].id,


         				        						      positionLastCustomer, planifiedCustomers[lastServedCustomer].pRouting.serviceTime,


         				        			                 customer.id, generalizedTour, positionGeneralizedTour, generalizedTourCost, commitCustomer);

 						*/
         			   }

         			 else

         			 {

         		      generalizedInsertionTypeOneAlgorithm (pRoutes[tour], GenRoute, tour, firstTimeServiceCurrentPosition[tour], customer.id, generalizedTour, positionGeneralizedTour, generalizedTourCost, commitCustomer);


         			//  generalizedInsertionTypeTwoAlgorithm (pRoutes[tour], GenRoute, tour, firstTimeServiceCurrentPosition[tour], customer.id, generalizedTour,positionGeneralizedTour, generalizedTourCost,  commitCustomer);

         		    }

         		}


         		 double costNewInsertionDepot = distance(dist,0 , customer.id) +  distance (dist, customer.id,0) ;


         		 double  dueTimeDepot = _tstep + _tslice +  costNewInsertionDepot;


         		 if( commitCustomer && costNewInsertionDepot < generalizedTourCost  && dueTimeDepot <= clients[0].durationService && pRoutes.size() <= FLEET_VEHICLES)


         			 commitCustomer = false;


         		 if (!commitCustomer)


         		 {

         			 if (pRoutes.size() <= FLEET_VEHICLES)

         			 {

         				 Route newRoute = emptyRoute();

         				 newRoute.push_back(customer.id);


         				 pRoutes.push_back(newRoute);


         				 firstTimeServiceCurrentPosition.push_back(_tstep + _tslice); ///////////////


         				 customer.pRouting.route = pRoutes.size();


         				 customer.pRouting.routePosition = pRoutes.back().size() ;


         			 }

         			 else


         				 customer.pRouting.route =customer.pRouting.routePosition = -1;

         		 }


         		 else

         		 {

       		      pRoutes[generalizedTour]= GenRoute;

            		  customer.pRouting.route = generalizedTour + 1;

            		  customer.pRouting.routePosition = positionGeneralizedTour + 1;


            		 for(size_t i = positionGeneralizedTour + 1 , size = pRoutes[generalizedTour].size() ; i < size; i++)

            		   {

            		      for (size_t j = 0 , sizeP = this -> size() ; j < sizeP ; j++)
            		           {

            		           	if(planifiedCustomers[j].id == pRoutes[generalizedTour][i])
            		          		   {

            		           		     planifiedCustomers[j].pRouting.routePosition = i +1;

            		           			 break ;

            		           	        }

            		       	   }

       			   }


         		}



         		 customer.velocity = randVelocity(pRoutes.size()-1);


         		 velocities.push_back(customer.velocity);


         		 this->push_back(customer.pRouting.route);


         		 customer.pRouting.is_served = false;


         		 customer.pRouting.serviceTime = -1;


         		 planifiedCustomers.push_back(customer);


         		 i = ( i + 1 ) % newCustomers.size();



         	 }


         	        invalidate ();

         	        invalidateBest();


      }





         routingInfo & eoParticleDVRP::currentRoutingCustomer(unsigned id_customer)
         {

        	 int i = 0;

        	 while (planifiedCustomers[i].id != id_customer) {++i;};


        	 if (i > planifiedCustomers.size())

        		 cerr<<"The current routing of the customer with id"<<id_customer<<"does not exist"<<endl;

        	 else

        		 return planifiedCustomers[i].pRouting ;

         }



         routingInfo & eoParticleDVRP::bestRoutingCustomer(unsigned id_customer)

         {

               	 int i = 0;

               	 while (planifiedCustomers[i].id != id_customer) {++i;};


               	 if (i > planifiedCustomers.size())

               		 cerr<<"The best routing of the customer with id"<<id_customer<< "does not exist"<<endl;

               	 else

               		 return planifiedCustomers[i].bestRouting ;

          }



       void eoParticleDVRP::commitOrdersCurrentPosition (double _nextTstep, double _timeSlice)

         {

    	   double dispoTime;

    	   unsigned lastServedCustomer,id_customer ;

        	 for (size_t tr = 0, size = pRoutes.size() ; tr < size ; tr++)
        	 {
        		 for (size_t i =1, sizeTr = pRoutes[tr].size(); i < sizeTr; i++)
        		 {
        			 id_customer = pRoutes[tr][i];

        			 if ( ! isServedCustomerInCurrentPosition (id_customer))

        			 {
        				 lastServedCustomer = pRoutes[tr][i-1];

        				 if (lastServedCustomer ==0) // depot
           				     dispoTime = firstTimeServiceCurrentPosition[tr];
        				 else
           					 dispoTime = timeEndServiceCurrentPosition(lastServedCustomer);

           				 if (dispoTime <= _nextTstep + _timeSlice){
           					 currentRoutingCustomer(id_customer).is_served = true ;
           					 currentRoutingCustomer(id_customer).serviceTime = startServiceTime(_nextTstep,dispoTime) +  distance (dist, lastServedCustomer, id_customer) ;
           					}

           				 else
           					break;
        			 }
        		 }
        	 }
         }

       void eoParticleDVRP::commitOrdersBestPosition (double _nextTstep, double _timeSlice)

               {
    	           double dispoTime ;
    	           unsigned id_customer,lastServedCustomer;

              	 for (size_t tr = 0, size = bestRoutes.size() ; tr < size ; tr++)
              	 {
              		 for (size_t i =1, sizeTr = bestRoutes[tr].size(); i < sizeTr; i++)
              		 {
              			 id_customer = bestRoutes[tr][i];

              			 if ( ! isServedCustomerInBestPosition (id_customer))
              		      {
              				 unsigned lastServedCustomer = bestRoutes[tr][i-1];

              				 if (lastServedCustomer ==0)
              					 dispoTime = firstTimeServiceBestPosition[tr];
              				 else
              					 dispoTime = timeEndServiceBestPosition(lastServedCustomer);

              				 if (dispoTime <=_nextTstep + _timeSlice)
              				 {
              					 bestRoutingCustomer(id_customer).is_served = true ;

              					 bestRoutingCustomer(id_customer).serviceTime = startServiceTime(_nextTstep,dispoTime) +  distance (dist, lastServedCustomer, id_customer);

              					               				 }
              				 else

           					 break;
           				 }
              		 }
              	 }
               }




         bool eoParticleDVRP::isServedCustomerInCurrentPosition (unsigned id_customer)
         {

        	 for (size_t i =0 ; i < planifiedCustomers.size(); i++)

       			 if (planifiedCustomers[i].id == id_customer )

       				 if(planifiedCustomers[i].pRouting.is_served)

       				    return true;



        	 return false;

         }


         bool eoParticleDVRP::isServedCustomerInBestPosition (unsigned id_customer)
                 {

                	 for (size_t i =0 ; i < planifiedCustomers.size(); i++)

               			 if (planifiedCustomers[i].id == id_customer)

               				 if(planifiedCustomers[i].bestRouting.is_served)

               				    return true;



                	 return false;

                 }




         void eoParticleDVRP::normalizeVelocities()
         	{

         	  if (velocities.size() != size())

         		  std::cerr<< " The Particle size is different to velocity size, in normalization process...!!!"<<endl;


         	  for(size_t i =0, size = this -> size(); i< size;++i)

         		  if(planifiedCustomers[i].pRouting.is_served)

         			  velocities[i] = planifiedCustomers[i].velocity =0;


            }



         void eoParticleDVRP::checkCurrentPosition()
         {

        	 for(size_t i =0, size = this -> size(); i< size;++i)

        		 cout<<endl<<planifiedCustomers[i].pRouting.route<<"("<<this ->operator[](i)<<")"<<'\t';

        	 cout<<endl;


         }



         void eoParticleDVRP::checkBestPosition()
                  {

                 	 for(size_t i =0, size = this -> size(); i< size;++i)

                 		 cout<<endl<<planifiedCustomers[i].bestRouting.route<<"("<<this ->operator[](i)<<")"<<'\t';

                 	 cout<<endl;


                  }




         void eoParticleDVRP::eraseRoute(unsigned  _tour)
         {


        	// cout<<"_tour  "<<_tour<<"  pRoutes.size()   "<< pRoutes.size()<<endl;

        	 pRoutes.erase(pRoutes.begin()+_tour);

        	 firstTimeServiceCurrentPosition.erase (firstTimeServiceCurrentPosition.begin() + _tour);


        	 for (unsigned tr = _tour , size = pRoutes.size() ; tr < size ; tr ++)
        	 {

        		 for (size_t  i = 1, sizetr = pRoutes[tr].size() ; i < sizetr ; i++)
        		 {

        			 unsigned customer = pRoutes[tr][i] ;


        			 for (size_t j = 0 , sizep = this->size() ; j < sizep ; j++ )


        				 if(planifiedCustomers[j].id == customer)

        					 {
        					  planifiedCustomers[j].pRouting.route -- ;

        			 		  this->operator[](j) --;

        			 		  break;

        					 }
        		 }


        	 }


         }



        bool  eoParticleDVRP::serviceIsProgressCurrentPosition(unsigned _tour)
        {


        	for(size_t i=1, sizetr = pRoutes[_tour].size() ; i < sizetr; i++)

        	{
        		unsigned customer = pRoutes[_tour][i];

        		for (size_t j = 0, size = this->size() ; j < size; j++)


        		    if(planifiedCustomers[j].id == customer)

        		    	 if (planifiedCustomers[j].pRouting.is_served)

        		        	return true ;
        		    	 else

        		    		 return false;


        	}

        	return false;


        }




        bool  eoParticleDVRP::serviceIsProgressBestPosition(unsigned _tour)
               {

            	   for(size_t i=1, sizetr = bestRoutes[_tour].size() ; i < sizetr; i++)

            	           	{
            	           		unsigned customer = bestRoutes[_tour][i];

            	           		for (size_t j = 0, size = this->size() ; j < size; j++)


            	           		    if(planifiedCustomers[j].id == customer)

            	           		    	 if (planifiedCustomers[j].bestRouting.is_served)

            	           		        	return true ;
            	           		    	 else

            	           		    		 return false;

            	           	}

            	   return false;


               }



        unsigned eoParticleDVRP::IndexLastServedCustomerCurrentPosition(unsigned _tour)
        {

        	unsigned customer,  index = 0 ;

        	for (size_t i = 1 , sizetr = pRoutes[_tour].size() ; i < sizetr ; i ++ )
        	{

        		customer = pRoutes[_tour][i];


        		for (unsigned j = 0 , size = this -> size(); j < size ; j ++)


        			if( planifiedCustomers[j].id == customer)

        				if (planifiedCustomers[j].pRouting.is_served)

        						index = j ;
        				else
        					return index ;
        	}
                 	return index ;
        }





        unsigned eoParticleDVRP::IndexLastServedCustomerBestPosition(unsigned _tour)
               {

            	   unsigned customer, index = 0 ;


            	   	for (size_t i = 1 , sizetr = bestRoutes[_tour].size() ; i < sizetr ; i ++ )
                    	{

       	           		 customer = bestRoutes[_tour][i];


       	           		for (unsigned j = 0 , size = this -> size(); j < size ; j ++)

            	           	if( planifiedCustomers[j].id == customer)

            	       			if (planifiedCustomers[j].bestRouting.is_served)

                					index = j ;

            	           	    else

            	           	    return index ;

            	         }

            	   return index ;

               }



       double eoParticleDVRP::timeEndServiceCurrentPosition(unsigned id_customer)
       {


    	   for(size_t i = 0, size = this->size() ; i < size ; i++)

    		   if (id_customer == planifiedCustomers[i].id)

    			   return (planifiedCustomers[i].pRouting.serviceTime + clients[id_customer].durationService);


    	   cerr<<"The customer with id " <<id_customer<< " not exist"<<endl;



       }


       double eoParticleDVRP::timeEndServiceBestPosition(unsigned id_customer)
             {
          	   for(size_t i = 0, size = this->size() ; i < size ; i++)

          		   if (id_customer == planifiedCustomers[i].id)

          			   return (planifiedCustomers[i].bestRouting.serviceTime+ clients[id_customer].durationService);


          	   cerr<<"The customer with id " <<id_customer<< " not exist"<<endl;


             }

       void eoParticleDVRP::copyBestToCurrent()
       {
    	   for(size_t i = 0, size = this->size(); i < size ; i ++)

    		   this->operator[](i)= bestPositions[i];


       }

       void eoParticleDVRP::reDesign ()
       {

    	   for(size_t tr = 0, size = pRoutes.size(); tr < size ; tr++)

    		   for(size_t i = 0, sizetr = pRoutes[tr].size() ; i < sizetr ; i++)
    		   {

    			   unsigned customer = pRoutes[tr][i] ;

    			   for (size_t j =0 , Psize = this -> size() ; j < Psize ; j ++)

    				   if (customer == planifiedCustomers[j].id)

    					   {

    					   planifiedCustomers[j].pRouting.route  = this -> operator [](j) = tr + 1 ;


    					    planifiedCustomers[j].pRouting.routePosition = i + 1 ;


    					   }

        		   }
           }


       void eoParticleDVRP::lastPositionOnRoutes(std::vector<unsigned> & _positions)
             {
          	   unsigned lastServedCustomer,positionLastCustomer;

          	  for(unsigned tour=0, size=pRoutes.size(); tour < size; tour++)
          	  {
          		  lastServedCustomer = IndexLastServedCustomerCurrentPosition(tour) ;

          		  positionLastCustomer = planifiedCustomers[lastServedCustomer].pRouting.routePosition -1 ;

          		  _positions.push_back(positionLastCustomer);

          	  }

             }



       void eoParticleDVRP::cleanParticle()
       {

    	   size_t i = 0;

    	   while( i < this ->size())

    	   {  if( ! planifiedCustomers[i].pRouting.is_served)
    		   {

    			   unsigned route = planifiedCustomers[i].pRouting.route - 1 ;

    			   unsigned position = planifiedCustomers[i].pRouting.routePosition - 1 ;

    			   pRoutes[route].erase(pRoutes[route].begin() + position);

    			   for(size_t j = position, sizetr = pRoutes[route].size(); j < sizetr; j++)

    				   for(size_t k = 0 , sizeP = this-> size() ; k < sizeP; k ++)

    				   {
    					   if (planifiedCustomers[k].id == pRoutes[route][j] )

    						     planifiedCustomers[k].pRouting.routePosition -- ;


    				   }


    			   if (pRoutes[route].size()==1)

    				   eraseRoute(route);


    			   newCustomers.push_back(planifiedCustomers[i].id);



    			   planifiedCustomers.erase( planifiedCustomers.begin() + i );

    			   this -> erase(this -> begin() + i);

    			   velocities.erase(velocities.begin() + i);

       			 continue;


    		   }

    	   i++;
    	   }

       }


       void eoParticleDVRP::checkAll(std::ostream& _os)
            {
          	   double  loadCharge , dueTime;

          	   loadCharge = dueTime = 0.0;

          	   _os<<'\t'<<"Last"<<'\t'<<"Capacity "<<VEHICULE_CAPACITY<<'\t'<<"TDay "<< TIME_DAY<<'\t'<<" Lenght"<<endl;

          	   for(size_t tr = 0, sizetr = bestRoutes.size(); tr < sizetr; tr ++)
          	   {
          		   dueTime = bestRoutingCustomer(bestRoutes[tr][bestRoutes[tr].size()-2]).serviceTime + clients[bestRoutes[tr][bestRoutes[tr].size()-2]].durationService + distance(dist,bestRoutes[tr][bestRoutes[tr].size()-2],0);

          		   loadCharge = getCapacityUsed(bestRoutes[tr]);

          		   _os<<"#"<<tr<<'\t'<< bestRoutes[tr][bestRoutes[tr].size()-2]<<'\t'<<loadCharge<<'\t'<<'\t'<<dueTime<<'\t'<<'\t'<< computTourLength(bestRoutes[tr])<<endl;

          	   }
          }



