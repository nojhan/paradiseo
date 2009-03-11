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
#ifndef EOPSODVRPFLIGHT_H_
#define EOPSODVRPFLIGHT_H_


#include "eoPsoDVRP.h"

#include "eoPsoDVRPutils.h"

#include "eoGlobal.h"




template < class POT >

class eoPsoDVRPMove:public eoFlight < POT >
{

public:
	 typedef typename POT::AtomType PositionType;


	 eoPsoDVRPMove() {}

	 void operator () (POT & _po)
	 {

		 POT _poSave = _po ;

		 deleteCustomersToMove(_po);

		 if (_po.pRoutes.size() != 0)
		 {

    		for (unsigned customer = 0, size = _po.size() ; customer < size; customer++)


    			 if ( ! (_po.planifiedCustomers[customer].pRouting.is_served))
    			 {

    				 unsigned xLimit = _po.pRoutes.size() -1 ;

    				 PositionType customer_route = _po[customer] -1 ;

    				 PositionType newTour = boundTour((customer_route + _po.velocities[customer]), xLimit);  //

    				 repositionCustomer(customer,newTour,_po);

    			 }
		 }
		 else
		 {

			 _po = _poSave;

			 for (unsigned customer = 0, size = _po.size() ; customer < size; customer++)

				 if ( ! (_po.planifiedCustomers[customer].pRouting.is_served))

				 {

					 unsigned xLimit = _po.pRoutes.size() -1;

					 PositionType customer_route = _po[customer] -1 ;

					 PositionType newTour =  boundTour((customer_route + _po.velocities[customer]), xLimit);

					repositionCustomerInFirst(customer,newTour,_po);

				 }

		 }
		 _po.invalidate();

	}



	 bool deleteCustomersToMove(POT & _po)
	       {


	    	   for(size_t i = 0; i < _po.size() ; i ++)
	    	   {
	    		   if( ! _po.planifiedCustomers[i].pRouting.is_served )
	    		   {

	    			   unsigned route = _po.planifiedCustomers[i].pRouting.route - 1 ;

	    			   unsigned position = _po.planifiedCustomers[i].pRouting.routePosition - 1 ;

	    			   _po.pRoutes[route].erase(_po.pRoutes[route].begin() + position);


	    			    if (_po.pRoutes[route].size()== 1)

	    			    	_po.eraseRoute(route);

	    			 else

	    			   for(size_t j = position, sizetr = _po.pRoutes[route].size(); j < sizetr; j++)

	    				   for(size_t k = 0 , sizeP = _po.size() ; k < sizeP; k ++)

	    				   {
	    					   if (_po.planifiedCustomers[k].id == _po.pRoutes[route][j] )

	    						     _po.planifiedCustomers[k].pRouting.routePosition -- ;


	    				   }

	    	          }
	       }

		}





	 /*
	   eoRealVectorBounds bounds(po.size(),0,xlimit);
	     if (bounds.isMinBounded(i))
		   newTour=std::max(newTour,bounds.minimum(i));
		   if (bounds.isMaxBounded(i))
			   newTour=std::min(newTour,bounds.maximum(i));

     */

	 PositionType boundTour(PositionType tour, unsigned limit)
	 {


		        if(limit == 0)

		           return(0);

		        else

		        	if (tour <0)

		        	    return(((-tour) % limit)) ;
		         	else

		         		if (tour > limit)

		        		   return ((tour % limit));

		 				else
		        		   return (tour);





	 }



	void repositionCustomer(unsigned _customer, unsigned _tour, POT&  _po)
	 {
		 Route newRoute;

			 double dueTime ;

			 bool commitCustomer ;

			 unsigned  positionNewCustomer;

			 double depotTimeWindow = clients[0].durationService;

			 unsigned  customer_id = _po.planifiedCustomers[_customer].id;


			 commitCustomer = false ;


			 for (size_t j = 0 ; j < _po.pRoutes.size() ; j++ )
			     {


				 if(_po.serviceIsProgressCurrentPosition(_tour))

					{


					 unsigned lastServedCustomer =_po.IndexLastServedCustomerCurrentPosition(_tour) ;


					 unsigned positionLastCustomer = _po.planifiedCustomers[lastServedCustomer].pRouting.routePosition ;


					 positionNewCustomer = randPosition(positionLastCustomer, _po.pRoutes[_tour].size());


					  dueTime  = getTimeOfService(_po.pRoutes[_tour],  _po.planifiedCustomers[lastServedCustomer].id, _po.planifiedCustomers[lastServedCustomer].pRouting.serviceTime, customer_id, positionNewCustomer)


					              + distance (dist, _po.pRoutes[_tour].back(),0);


					}

				 else


					 {

					 positionNewCustomer = randPosition(1, _po.pRoutes[_tour].size());


					 dueTime  = getTimeOfService(_po.pRoutes[_tour],_po.firstTimeServiceCurrentPosition[_tour], customer_id, positionNewCustomer)+ distance (dist, _po.pRoutes[_tour].back(),0);

					 }



			        double demandTour =  getCapacityUsed(_po.pRoutes[_tour]) + clients[customer_id].demand;


			        if (dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )

			           {


			        	_po.pRoutes[_tour].insert(_po.pRoutes[_tour].begin() + positionNewCustomer, customer_id );

			            _po.planifiedCustomers[_customer].pRouting.route =  _po[_customer] =  _tour+1;

			            _po.planifiedCustomers[_customer].pRouting.routePosition = positionNewCustomer +1 ;



			            for(size_t i = positionNewCustomer +1 , size = _po.pRoutes[_tour].size() ; i < size; i++)


			            {
			            	for (size_t j = 0 , sizeP = _po.size() ; j < sizeP ; j++)

			            	{
			            		if(_po.planifiedCustomers[j].id == _po.pRoutes[_tour][i])

			            		{
			            			_po.planifiedCustomers[j].pRouting.routePosition ++;


			            			break ;


			            		}


			            	}


			            }



			            commitCustomer = true ;

			   	        break;

			   		   }

			        _tour = ( _tour + 1 ) % _po.pRoutes.size();

	 		    }

			     if (!commitCustomer)

			          if (_po.pRoutes.size() < FLEET_VEHICLES)

			          {

			         	    Route newRoute = emptyRoute();

			                newRoute.push_back(customer_id);

		 		   	       _po.pRoutes.push_back(newRoute);

		 		   	       _po.firstTimeServiceCurrentPosition.push_back(TIME_STEP);// + TIME_SLICE);

		 		   	       _po.planifiedCustomers[_customer].pRouting.route = _po[_customer] = _po.pRoutes.size();

		 		   	       _po.planifiedCustomers[_customer].pRouting.routePosition = _po.pRoutes.back().size() ;

			          }

			          else

			          _po.planifiedCustomers[_customer].pRouting.route =_po[_customer] = _po.planifiedCustomers[_customer].pRouting.routePosition = -1;

		 }



	void repositionCustomerInFirst(unsigned _customer, unsigned _tour, POT&  _po)
	 {

		 Route newRoute;

		 double dueTime ;

		 bool commitCustomer ;

		 unsigned  positionNewCustomer;

		 double depotTimeWindow = clients[0].durationService;

		 unsigned  customer_id = _po.planifiedCustomers[_customer].id;

		 unsigned  route = _po[_customer] -1  ;

		 unsigned position = _po.planifiedCustomers[_customer].pRouting.routePosition -1;


		 _po.pRoutes[route].erase(_po.pRoutes[route].begin() + position);



		 for ( unsigned i = position, size = _po.pRoutes[route].size(); i < size; i++)   ///en premier

			{ 	unsigned upcustomer = _po.pRoutes[route][i];

				for (size_t j = 0, Psize = _po.planifiedCustomers.size(); j < Psize ; j++)

		 			if (_po.planifiedCustomers[j].id == upcustomer)

		 				{ _po.planifiedCustomers[j].pRouting.routePosition --;


		 				  break;
		 				}
			}


		 if(_po.pRoutes[route].size() == 1 )//&& _po.pRoutes[route].size() >1)  	 //empty tour, only the depot

			{ _po.eraseRoute(route);

			    unsigned xLimit = _po.pRoutes.size() -1 ;

			   _tour = boundTour(_tour,xLimit);


			}




		 commitCustomer = false ;


		 for (size_t j = 0 ; j < _po.pRoutes.size() ; j++ )
		     {


			 if(_po.serviceIsProgressCurrentPosition(_tour))

				{


				 unsigned lastServedCustomer =_po.IndexLastServedCustomerCurrentPosition(_tour) ;


				 unsigned positionLastCustomer = _po.planifiedCustomers[lastServedCustomer].pRouting.routePosition ;


				 positionNewCustomer = randPosition(positionLastCustomer, _po.pRoutes[_tour].size());


				  dueTime  = getTimeOfService(_po.pRoutes[_tour],  _po.planifiedCustomers[lastServedCustomer].id, _po.planifiedCustomers[lastServedCustomer].pRouting.serviceTime, customer_id, positionNewCustomer)


				              + distance (dist, _po.pRoutes[_tour].back(),0);


				}

			 else


				 {

				 positionNewCustomer = randPosition(1, _po.pRoutes[_tour].size());


				 dueTime  = getTimeOfService(_po.pRoutes[_tour],_po.firstTimeServiceCurrentPosition[_tour], customer_id, positionNewCustomer)+ distance (dist, _po.pRoutes[_tour].back(),0);

				 }



		        double demandTour =  getCapacityUsed(_po.pRoutes[_tour]) + clients[customer_id].demand;


		        if (dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )

		           {


		        	_po.pRoutes[_tour].insert(_po.pRoutes[_tour].begin() + positionNewCustomer, customer_id );

		            _po.planifiedCustomers[_customer].pRouting.route =  _po[_customer] =  _tour+1;

		            _po.planifiedCustomers[_customer].pRouting.routePosition = positionNewCustomer +1 ;



		            for(size_t i = positionNewCustomer +1 , size = _po.pRoutes[_tour].size() ; i < size; i++)


		            {
		            	for (size_t j = 0 , sizeP = _po.size() ; j < sizeP ; j++)

		            	{
		            		if(_po.planifiedCustomers[j].id == _po.pRoutes[_tour][i])

		            		{
		            			_po.planifiedCustomers[j].pRouting.routePosition ++;


		            			break ;


		            		}


		            	}


		            }



		            commitCustomer = true ;

		   	        break;

		   		   }

		        _tour = ( _tour + 1 ) % _po.pRoutes.size();

 		    }

		     if (!commitCustomer)

		          if (_po.pRoutes.size() < FLEET_VEHICLES)

		          {

		         	    Route newRoute = emptyRoute();

		                newRoute.push_back(customer_id);

	 		   	       _po.pRoutes.push_back(newRoute);

	 		   	       _po.firstTimeServiceCurrentPosition.push_back(TIME_STEP );//+ TIME_SLICE);

	 		   	       _po.planifiedCustomers[_customer].pRouting.route = _po[_customer] = _po.pRoutes.size();

	 		   	       _po.planifiedCustomers[_customer].pRouting.routePosition = _po.pRoutes.back().size() ;

		          }

		          else

		          _po.planifiedCustomers[_customer].pRouting.route =_po[_customer] = _po.planifiedCustomers[_customer].pRouting.routePosition = -1;


	 }




};



#endif /*EOPSODVRPFLIGHT_H_*/
