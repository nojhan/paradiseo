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


#include "eoPsoDVRPutils.h"


   // The computed distances will be stored in dist.

setCustumers clients;

std::vector<unsigned> newCustomers;

lengthMatrix dist;


template<typename T>

std::string to_string( const T & Value )
{
    // utiliser un flux de sortie pour créer la chaîne

	std::ostringstream oss;

	// écrire la valeur dans le flux

	oss << Value;

	// renvoyer une string
    return oss.str();
}






void computeDistances() {


     dist.resize (clients.size ()) ;

     for (unsigned i = 0; i < dist.size (); i ++)

    	 dist [i].resize (clients.size ());

     // Distances computation
     for (unsigned i = 0; i < dist.size (); i ++)

    	 for (unsigned j = i  ; j < dist.size (); j ++)

    	 {

             double distX = clients [i].x - clients [j].x;

             double distY = clients [i].y - clients [j].y;

             dist [i][j] = dist [j][i] = sqrt (distX * distX + distY * distY);

      }
}

/**
   * \brief Returns the time window information of a given client.
   * \param _client The client whose information we want to know.
   * \param _availTime Return value. The time xhen client will be available in system.
   * \param _durationService Return value. The client's service time.
   */

void getInformation (unsigned _custumer, double& _availTime, double& _durationService) {

    assert (_custumer >= 0 && _custumer < clients.size ());

   _availTime = clients [_custumer].availTime;

   _durationService = clients [_custumer].durationService;

}



/**
   * \brief A function to get the distance between two clients.
   * \param _from The first client.
   * \param _to The second client.
   * \return The distance between _from and _to.
   */

void PrintCustumersMatrix (){

	for (int i =0 ; i < clients.size(); ++i)

		cout<< clients[i].id<<'\t'<<clients[i].x<<'\t'<<clients[i].y<<'\t'<<clients[i].demand<<'\t'<<clients[i].durationService<<endl;

}


double distance (const lengthMatrix _dist, unsigned _from, unsigned  _to) {

    assert (_from >= 0 && _from < clients.size ());

    assert (_to   >= 0 && _to   < clients.size ());

    return _dist [_from][_to];

}


const lengthMatrix  distanceMatrix() {return dist;}


/**
   * \brief Loads the problem data from a given file.
   * \param _fileName The file to load data from.
   * \warning No error check is performed!
   */


double computTourLength(Route _route)

            {
                double RouteLength =0.0;
                for (unsigned i = 0, size = _route.size()-1; i < size; ++i )
                   	RouteLength+=distance(dist,_route[i], _route[i+1]);

                RouteLength+= distance(dist, _route.back(),0);
                return RouteLength;

            }



double computTourLength( Routes _mRoute)

            {
                double MatrixLength =0.0;

                for(unsigned tr = 0, tsize = _mRoute.size(); tr < tsize; ++tr )

                     MatrixLength += computTourLength(_mRoute[tr]);

                return MatrixLength;
            }




bool findClient (unsigned id)  // Look for one customers in the table Clients

{
	for ( size_t i = 0, size = clients.size(); i < size; ++i )
    {
		if (id == (clients.at(i)).id)
			return true;
    }

	 return false;
}


void loadNewCustomers(double _timeStep, double _timeSlice){

	unsigned  id_customer;

	newCustomers.clear();

	unsigned i = 1;

	while (i < clients.size())
	{
		if(clients[i].availTime > (_timeStep - _timeSlice)  &&  clients[i].availTime <=_timeStep)

			newCustomers.push_back(clients[i].id);

		i++;

	}

}


void loadMatrixCustomers (const string _fileName) {

	string line;

	istringstream  iss;

	std::ifstream file (_fileName.c_str());

	ClientDataT depot, custumer;

	unsigned int id;


    if (file) {

            depot.id= 0;

            depot.availTime = depot.demand=depot.durationService = 0.0;

            clients.push_back (depot);


            do
             	getline(file,line);

            while (line.find("DEMAND_SECTION")== string::npos);

            getline(file,line);

            do{

                iss.str(line);

            	iss>> custumer.id>>custumer.demand;

            	clients.push_back (custumer);

            	iss.clear();

            	getline(file,line);

            }while(line.find("LOCATION_COORD_SECTION")== string::npos);


            	 getline(file,line);
            do{

            	iss.str(line);

            	iss>>id;

            	iss>>clients[id].x;

            	iss>>clients[id].y;

            	iss.clear();

            	getline(file,line);

           	  } while(line.find("DEPOT_LOCATION_SECTION")== string::npos);


            do
           		getline(file,line);


            while (line.find("DURATION_SECTION")== string::npos);


            	getline(file,line);


            do{    iss.str(line);

            	   iss>>id;

            	   iss>>clients[id].durationService;

            	   iss.clear();

            	   getline(file,line);

             }  while(line.find("DEPOT_TIME_WINDOW_SECTION")== string::npos);


            getline(file,line);

            iss.str(line);

            iss>>id;

            iss>>clients[id].availTime>>clients[id].durationService;

            iss.clear();

            do
            	getline(file,line);



            while(line.find("TIME_AVAIL_SECTION")== string::npos);


            	 getline(file,line);


            do{
            	iss.str(line);

            	iss>>id;

            	iss>>clients[id].availTime;

            	iss.clear();

            	getline(file,line);

             } while(line.find("EOF")== string::npos);



        file.close ();

     computeDistances();

    }
    else {

        std::cerr << "Error: the file: " << _fileName << " doesn't exist !!!" << std::endl ;

        abort();

         }

}


/**
  * \brief Prints a route to the standard output.
  * \param _route The route to print.
  */

void printRoute (const Route& _route, std::ostream &_os) {

    _os<< "[";

    for (unsigned i = 0; i < _route.size (); i++) {

        _os << _route [i];

        if (i != _route.size () -1)

        	_os << ", ";

    		}

    _os << "]"<<endl;

}


/**
  * \brief Prints a set of routes to the standard output.
  * \param _routes The set of routes to print.
  */

void printRoutes (const Routes& _routes, std::ostream &_os) {


    for (unsigned i = 0; i < _routes.size (); i++)

       {_os <<i+1<<'\t';

    	printRoute ( _routes [i],_os);
       }
}



double getTimeOfService (Route _route, double _startTime, unsigned _newCustomer_id, unsigned _newPosition)
{

	double duration = _startTime;


	if( _route.empty())

		cerr<< "You want to get the time of service from empty tour....!!!!"<<endl;


	_route.insert (_route.begin() + _newPosition, _newCustomer_id ) ;

	for (size_t i = 0 ; i  < _route.size()-1 ; ++ i )

		duration +=  distance(dist,_route[i], _route[i+1]);


	for (size_t i = 1 ; i  < _route.size() ; ++ i )  //depot

		duration += clients[_route[i]].durationService;


	return duration;

}

double getTimeOfService(Route _route, unsigned _lastServedCustomer_id, double _serviceTime, unsigned _newCustomer_id, unsigned _newPosition)
{

	double nextTimeStep = TIME_STEP; //

	double dispo =  _serviceTime + clients[_lastServedCustomer_id].durationService;

	double duration = startServiceTime( nextTimeStep , dispo );

	_route.insert(_route.begin() + _newPosition, _newCustomer_id);

	unsigned index = 1 ;

	while (_route[index] != _lastServedCustomer_id)	index ++ ;



	for (unsigned i = index; i  < _route.size() - 1 ; ++ i )

		duration+=  distance(dist,_route[i], _route[i+1]);


	for (unsigned i = index + 1 ; i  < _route.size() ; ++ i )

		duration+= clients[_route[i]].durationService ;


	return duration ;


}



double getTimeOfService(const Route _route, unsigned _lastServedCustomer_id, double dispo)

{


	double nextTimeStep = TIME_STEP ;

	double duration = startServiceTime( nextTimeStep , dispo );

	unsigned index = 1 ;

	while (_route[index] != _lastServedCustomer_id)	index ++ ;

	for (unsigned i = index; i  < _route.size() - 1 ; ++ i )


		duration+=  distance(dist,_route[i], _route[i+1]);


	for (unsigned i = index + 1 ; i  < _route.size() ; ++ i )

		duration+= clients[_route[i]].durationService ;


	return duration ;

}





double startServiceTime (double _nextTimeStep, double _dispoTime)
{

	if (_nextTimeStep > _dispoTime)


		return _nextTimeStep ;



	else

		return _dispoTime;

}




double getCapacityUsed (Route _route)

   {

	double charge =0.0;

   	unsigned id_custumer;

   for (size_t i=1, size = _route.size(); i <size ; ++i )
   	{
   		id_custumer =_route[i];

   		charge+= clients [id_custumer].demand;

     }

   	return charge;
   }


void PrintCustumers() {

	   for ( size_t i = 0, size = clients.size(); i < size; ++i )

		   std::cout<< clients[i].id <<'\t'

		   			<<clients[i].availTime<<'\t'

		   			<<clients[i].demand<<'\t'

		   			<<clients[i].x<<'\t'

		   			<<clients[i].y<<'\t'

		   			<<clients[i].durationService <<endl;


	   for ( size_t i = 0, size = dist.size(); i < size; ++i )
	   	    {
		       std::cout<<endl;

		     for(size_t j = 0, size = dist.size(); j < size; ++j )

		      std::cout<< dist[i][j]<<'\t';
	   	    }
}



void PrintLastCustomers(){


	std::cout<<"The last customers :"<<newCustomers.size()<<endl;

	for ( size_t i = 0, size = newCustomers.size(); i < size; ++i )

		std::cout<<newCustomers[i]<<'\t';



}

const setCustumers SetKnowClients(){

	return clients;

	}


Route emptyRoute (){

	Route newRoute;

	newRoute.push_back(0);

	return newRoute;

}


unsigned  randTour(unsigned maxTour)

	{
      	 eoUniformGenerator<unsigned int > uGen(0,maxTour);

       	 return uGen();
     }


int randVelocity(int sizetr)
	{



	eoUniformGenerator<int > uGen(-sizetr,sizetr);

	 return uGen();


	}



unsigned randPosition (unsigned _lastPosition, unsigned _sizeTour)

{

	eoUniformGenerator<unsigned> uGen(_lastPosition ,_sizeTour);

	return uGen();


}


void checkDistances(const Routes _matrix)

{

	double longeurTotal = 0.0 ;

	 for(size_t tr = 0, tsize =_matrix.size(); tr < tsize; ++tr )
	 {
		 double longeur = 0.0 ;

	    for (size_t i = 0, size = _matrix[tr].size()-1; i < size; ++i )

	      longeur +=distance(dist,_matrix[tr][i], _matrix[tr][i+1]);

		  cout << endl<<"distance route "<< tr+1 << "   "<< longeur <<endl ;

		  longeurTotal += longeur;
	 }

	 cout << endl<<"distance totale "<<  longeurTotal  << endl ;



}



void checkCharge(const Routes _matrix)
{


	 for(size_t tr = 0, tsize =_matrix.size(); tr < tsize; ++tr )

		 { double charge = 0.0 ;

		 for(size_t i = 1, size = _matrix[tr].size() ; i < size; ++i )

	      charge += clients [_matrix[tr][i]].demand;


		  cout << endl<<"charge of route "<< tr+1 << "   "<< charge <<endl ;
		 }
}


void visualization (const Routes _matrix, unsigned _seed, std::ofstream& _os)
{


	//ofstream file ("visualization.txt");

	 if ( !_os )
	    {
	        cerr << "Creation file failed\n";
	        return;
	    }

  _os<<"Seed   "<<_seed<<endl;

  _os<<endl<<"#index 0"<<endl;

  _os<<clients[0].x<<'\t'<< clients[0].y<<endl;

  _os<<endl<<endl<<"#index 1"<<endl;

	 for (size_t i = 1, size = clients.size(); i < size; i++)

		 _os << clients[i].id<<'\t'<<clients[i].x<<'\t'<<clients[i].y<<endl;


	for (size_t i =0, sizeL = _matrix.size(); i < sizeL ; i++)
	{

		_os<<endl<<endl<<"#index "<<i+2<<endl;

		for(size_t j = 0, sizeC = _matrix[i].size(); j < sizeC ; j++)


			_os << _matrix[i][j]<<'\t'<< clients[_matrix[i][j]].x<<'\t'<< clients[_matrix[i][j]].y<<endl;


	}

	_os <<endl<<"--********************************************************************************"<<endl;


	_os.close();

}




void cheapestInsertionAlgorithm (Route _route, unsigned indexTour,

								double _startTime, unsigned customer_id, unsigned & cheapestTour,double & cheapestTourCost,

								unsigned & positionCheapestTour, bool & commitCustomer)

{

	double costInsertion, cheapestCostInsertion, depotTimeWindow = clients[0].durationService;

	unsigned positionTour, cheapestPosition;

	bool  feasibleInsertion = false ;

	for( size_t i = 1, sizetr = _route.size()-1  ; i < sizetr; i ++)
	{

		if ( i < _route.size() - 1)
		{

		unsigned lastCustomer_id = _route [i];

		unsigned nextCustomer_id = _route[i+1];

		 costInsertion = distance(dist,lastCustomer_id , customer_id) +

			             distance (dist, customer_id, nextCustomer_id) -

			             distance (dist, lastCustomer_id, nextCustomer_id);
		}
		else

		 costInsertion = distance (dist, _route.back(), customer_id )  +

		 				   distance (dist, customer_id, 0) -

		 				distance (dist, _route.back(), 0);



		 positionTour =  i+1 ;


		double  dueTime  = getTimeOfService(_route,_startTime, customer_id, positionTour)


		  			     +  distance (dist,_route.back(),0) ;



	   double demandTour =  getCapacityUsed(_route) + clients[customer_id].demand;

	   if (dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )

	   	        {

	   	    	  if ((!feasibleInsertion) || (costInsertion < cheapestCostInsertion))

	   	    	   {  cheapestCostInsertion = costInsertion;

	   	    	     cheapestPosition = positionTour;

	   	    	     feasibleInsertion = true;


	   	    	   }

	   	       }
	}


	if(!commitCustomer && feasibleInsertion)
			{

			  commitCustomer = true;

			  cheapestTourCost  = cheapestCostInsertion;

			  positionCheapestTour =  cheapestPosition;

			  cheapestTour = indexTour;

			}

	else
	{

		if(feasibleInsertion)

			if( cheapestCostInsertion < cheapestTourCost )

	   		{
	   		  cheapestTourCost  = cheapestCostInsertion;

	   		  positionCheapestTour =  cheapestPosition;

	   		  cheapestTour = indexTour;


	   		}


	}
}




void cheapestInsertionAlgorithm (Route _route, unsigned indexTour,

								 unsigned lastServedCustomer_id, unsigned lastServedCustomerPosition,

								 double serviceTime, unsigned customer_id, unsigned & cheapestTour,

								 double & cheapestTourCost, unsigned & positionCheapestTour, bool & commitCustomer)
{

	double costInsertion, cheapestCostInsertion, depotTimeWindow = clients[0].durationService;

	unsigned positionTour, cheapestPosition;

	bool  feasibleInsertion = false ;


	for( size_t i = lastServedCustomerPosition  +1  , sizetr = _route.size() - 1  ; i <  sizetr; i ++)

	{

		if(i < _route.size() - 1 )

		{

		unsigned lastCustomer_id = _route [i];

	    unsigned nextCustomer_id = _route[i+1];


	    costInsertion = distance(dist, lastCustomer_id, customer_id) +

	                    distance (dist, customer_id, nextCustomer_id) -

	                    distance (dist, lastCustomer_id, nextCustomer_id);
		}
		else


		costInsertion = distance (dist, _route.back(), customer_id) +  /////

						distance (dist, customer_id, 0)-

		 				distance (dist, _route.back(), 0);


	    positionTour =  i + 1 ;

	    double dueTime  =  getTimeOfService(_route, lastServedCustomer_id, serviceTime, customer_id, positionTour)

          	   	   			+ distance (dist, _route.back(),0);


	    double demandTour =  getCapacityUsed(_route) + clients[customer_id].demand;


	       if (dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )

	        {

	    	  if ((!feasibleInsertion) || (costInsertion < cheapestCostInsertion))

	    	   {  cheapestCostInsertion = costInsertion;

	    	      cheapestPosition = positionTour;

	    	      feasibleInsertion = true;

	    	   }

	       }
  }




	if(!commitCustomer && feasibleInsertion)
			{

			  commitCustomer = true;

			  cheapestTourCost  = cheapestCostInsertion;

			  positionCheapestTour =  cheapestPosition;

			  cheapestTour = indexTour;

			}

	else
	{

		if(feasibleInsertion)

			if( cheapestCostInsertion < cheapestTourCost )

	   		{
	   		  cheapestTourCost  = cheapestCostInsertion;

	   		  positionCheapestTour =  cheapestPosition;

	   		  cheapestTour = indexTour;


	   		}


	}

}




void nearestInsertionAlgorithm (Route _route, unsigned indexTour,

								double _startTime, unsigned customer_id, unsigned & nearestTour,double & nearestTourCost,

								unsigned & positionNearestTour, bool & commitCustomer)
{

	double costInsertion, nearestCostInsertion, nearestCheapCostInsertion , costLeftInsertion , costRightInsertion;

	unsigned leftCustomer_id, rightCustomer_id, nearestPosition, positionTour;

	bool feasibleInsertion = false ;

	double depotTimeWindow = clients[0].durationService;

	for( size_t i = 0, sizetr = _route.size() ; i < sizetr ; i ++)
	{

		unsigned nearestCustomer_id = _route [i];

		costInsertion = distance (dist, nearestCustomer_id, customer_id);

		if (((!feasibleInsertion) ) || ( costInsertion < nearestCostInsertion))
		{

			if( nearestCustomer_id != 0 )
			{

				leftCustomer_id  =  _route[i-1];


				if(i == _route.size()-1)

					rightCustomer_id = 0;

				else

    				rightCustomer_id = _route[i+1];

				costLeftInsertion = distance (dist , leftCustomer_id , customer_id) +

								    distance (dist, customer_id, nearestCustomer_id ) -

								    distance (dist, leftCustomer_id, nearestCustomer_id ) ;

				costRightInsertion = distance(dist, nearestCustomer_id, customer_id) +

									 distance(dist, customer_id , rightCustomer_id) -

									 distance (dist , nearestCustomer_id , rightCustomer_id);


				if (costLeftInsertion < costRightInsertion)

				  positionTour = i ;

			    else

				  positionTour  = i + 1 ;

				}
			else

				positionTour = i + 1 ;


				double  dueTime  = getTimeOfService(_route, _startTime, customer_id, positionTour)

						  			     +  distance (dist,_route.back(),0) ;


				double demandTour =  getCapacityUsed(_route) + clients[customer_id].demand;



				if( dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )


					{
					   nearestCostInsertion =costInsertion ;

				       nearestPosition = positionTour;

					  feasibleInsertion = true;

				    }
				else

					if( nearestCustomer_id !=0  )

					{

						if( positionTour > i)


							 positionTour = i;

						 else

							 positionTour = i + 1;


				     double  dueTime  = getTimeOfService(_route, _startTime, customer_id, positionTour)

										  			     +  distance (dist,_route.back(),0) ;


					 double demandTour =  getCapacityUsed(_route) + clients[customer_id].demand;


					 if( dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )


					 		{
					 		   nearestCostInsertion = costInsertion ;

					 		    nearestPosition = positionTour;

					 			feasibleInsertion = true;

						    }

					}
		        }
         }





	if(!commitCustomer && feasibleInsertion)
			{

			  commitCustomer = true;

			  nearestTourCost  = nearestCostInsertion;

			  positionNearestTour =  nearestPosition;

			  nearestTour = indexTour;

			}

	else
	{

		if(feasibleInsertion)

			if(  nearestTourCost < nearestCostInsertion )

	   		{
	   		  nearestTourCost  = nearestCostInsertion;

	   		  positionNearestTour =  nearestPosition;

	   		  nearestTour = indexTour;


	   		}


	}

   }




void nearestInsertionAlgorithm (Route _route, unsigned indexTour,

								 unsigned lastServedCustomer_id, unsigned lastServedCustomerPosition,

								 double serviceTime, unsigned customer_id, unsigned & nearestTour,

								 double & nearestTourCost, unsigned & positionNearestTour, bool & commitCustomer)
{

	   double costInsertion, nearestCostInsertion, nearestCheapCostInsertion , costLeftInsertion , costRightInsertion;

		unsigned leftCustomer_id, rightCustomer_id, nearestPosition, positionTour;

		bool feasibleInsertion = false ;

		double depotTimeWindow = clients[0].durationService;

		for( size_t i = lastServedCustomerPosition, sizetr = _route.size() ; i < sizetr ; i ++)
		{

			unsigned nearestCustomer_id = _route [i];

			costInsertion = distance (dist, nearestCustomer_id, customer_id);

			if (((!feasibleInsertion) ) || ( costInsertion < nearestCostInsertion))
			{

				if( nearestCustomer_id != lastServedCustomer_id )
				{

					leftCustomer_id  =  _route[i-1];

					if(i == _route.size()-1)

						rightCustomer_id = 0;
					else

						rightCustomer_id = _route[i+1];



					costLeftInsertion = distance (dist , leftCustomer_id , customer_id) +

									    distance (dist, customer_id, nearestCustomer_id ) -

									    distance (dist, leftCustomer_id, nearestCustomer_id ) ;

					costRightInsertion = distance(dist, nearestCustomer_id, customer_id) +

										 distance(dist, customer_id , rightCustomer_id) -

										 distance (dist , nearestCustomer_id , rightCustomer_id);


					if (costLeftInsertion < costRightInsertion)

					  positionTour = i  ;

				    else

					  positionTour  = i + 1 ;

			 }
				else

					positionTour = i + 1 ;


				   double dueTime  =  getTimeOfService(_route, lastServedCustomer_id, serviceTime, customer_id, positionTour)

				          	   	   			+ distance (dist, _route.back(),0);

					double demandTour =  getCapacityUsed(_route) + clients[customer_id].demand;



					if( dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )


						{
						   nearestCostInsertion =costInsertion ;

					       nearestPosition = positionTour;

						  feasibleInsertion = true;

					    }
					else

						if( nearestCustomer_id !=lastServedCustomer_id)

						{


							if( positionTour > i)


								 positionTour = i;

							 else

								 positionTour = i + 1;


					      double dueTime  =  getTimeOfService(_route, lastServedCustomer_id, serviceTime, customer_id, positionTour)

							          	   	   			+ distance (dist, _route.back(),0);

						 double demandTour =  getCapacityUsed(_route) + clients[customer_id].demand;


						 if( dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )


						 		{
						 		   nearestCostInsertion = costInsertion ;

						 		    nearestPosition = positionTour;

						 			feasibleInsertion = true;

							    }

						}
			        }
	         }


		if(!commitCustomer && feasibleInsertion)
				{

				  commitCustomer = true;

				  nearestTourCost  = nearestCostInsertion;

				  positionNearestTour =  nearestPosition;

				  nearestTour = indexTour;

				}

		else
		{

			if(feasibleInsertion)

				if(nearestTourCost < nearestCostInsertion )

		   		{
		   		  nearestTourCost  = nearestCostInsertion;

		   		  positionNearestTour =  nearestPosition;

		   		  nearestTour = indexTour;


		   		}

		}

	  }




void generalizedInsertionTypeOneAlgorithm (Route _route, Route & _GenRoute,	unsigned indexTour, double _startTime, unsigned customer_id, unsigned & generalizedTour,

										unsigned & positionGeneralizedTour, double & generalizedTourCost, bool & commitCustomer)
{

	double costInsertion, lowCostInsertion;

	unsigned positionTour, lowPosition;

	std::vector <unsigned> tempNodes;

	bool  feasibleInsertion = false ;

	double depotTimeWindow = clients[0].durationService;

	Route _tempRoute, _GenTempRoute ;

	if(_route.size() >= 3)

	{

		for( size_t i =  1, sizetri = _route.size()-1; i < sizetri; i ++)
	   {

			for( size_t j =  i, sizetrj = _route.size()-1 ; j < sizetrj; j ++)


				for( size_t k = j+1, sizetrk = _route.size(); k < sizetrk ; k++)

			      {

					costInsertion = distance(dist,_route[i] , customer_id) + distance (dist, customer_id, _route[j]) + distance (dist, _route[i+1],_route[k]) + distance(dist,_route[j+1],_route[k+1])

						-

						distance (dist, _route[i],_route[i+1]) + distance (dist, _route[j], _route[j+1]) + distance(dist,_route[k],_route[k+1]);


    	        positionTour = i+1;


    	        _tempRoute = _route ;


    	        tempNodes.clear() ;


    	        tempNodes.push_back(j); tempNodes.push_back(k);


    	        InsertionTypeOne(_tempRoute,positionTour,customer_id,tempNodes);


    	        double dueTime = getTimeOfService(_tempRoute, 0, _startTime ) + distance (dist,_tempRoute.back() , 0);


    	        double demandTour =  getCapacityUsed(_tempRoute);


    	        if (dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )
	   	        {

    	        	if ((!feasibleInsertion) || (costInsertion < lowCostInsertion))

	   	    	   {

	   	    		  lowCostInsertion= costInsertion;

	   	    		  lowPosition = positionTour;

	   	    	      _GenTempRoute = _tempRoute;

	   	    	      feasibleInsertion = true;

	   	    	   }

	   	       }

			}
		}
	}
else
  {
	_tempRoute = _route ;

	positionTour = _tempRoute.size();

	_tempRoute.push_back(customer_id);

	costInsertion = distance(dist,_tempRoute.back(), customer_id) + distance(dist,customer_id, 0) - distance(dist,_tempRoute.back(),0) ;

	 double dueTime = getTimeOfService(_tempRoute, 0, _startTime ) + distance (dist,_tempRoute.back() , 0);

    double demandTour =  getCapacityUsed(_tempRoute) + clients[customer_id].demand;

    if (dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )

   	   {

    	if ((!feasibleInsertion) || (costInsertion < lowCostInsertion))

    		{

        	 lowCostInsertion= costInsertion;

    		 lowPosition = positionTour;

    		 _GenTempRoute = _tempRoute;

    		 feasibleInsertion = true;

    		}

   	   }

  }


if(feasibleInsertion)

	if( !commitCustomer || lowCostInsertion < generalizedTourCost )

	{
		generalizedTourCost  = lowCostInsertion;

		positionGeneralizedTour =  lowPosition;

		_GenRoute = _GenTempRoute;

		generalizedTour = indexTour;

		commitCustomer = true;


	}


}





void generalizedInsertionTypeOneAlgorithm (Route _route, Route& _Genroute, unsigned indexTour,  unsigned lastServedCustomer_id,

										   unsigned lastServedCustomerPosition, double serviceTime, unsigned customer_id, unsigned & generalizedTour,

										   unsigned &  positionGeneralizedTour, double & generalizedTourCost,   bool & commitCustomer)
{

	double costInsertion, lowCostInsertion;

	unsigned positionTour, lowPosition;

	std::vector <unsigned> tempNodes;

	bool  feasibleInsertion = false ;

	double depotTimeWindow = clients[0].durationService;

	Route 	_tempRoute,  _GenTempRoute;


	if(_route.size() >= 3)
	{

		for( size_t i = lastServedCustomer_id + 1, sizetri = _route.size() -1; i < sizetri; i ++)
		{

		for( size_t j =  i, sizetrj = _route.size() -1  ; j < sizetrj; j ++)


		for( size_t k = j+1, sizetrk = _route.size(); k < sizetrk ; k++)


		  {


			costInsertion = distance(dist,_route[i] , customer_id) + distance (dist, customer_id, _route[j]) + distance (dist, _route[i+1],_route[k]) + distance(dist,_route[j+1],_route[k+1])

						-

						distance (dist, _route[i],_route[i+1]) + distance (dist, _route[j], _route[j+1]) + distance(dist,_route[k],_route[k+1]);


			 positionTour = i+1;


		  	_tempRoute = _route ;


		  	 tempNodes.clear() ;


		  	 tempNodes.push_back(j);


		  	 tempNodes.push_back(k);


		  	 InsertionTypeOne(_tempRoute,positionTour,customer_id,tempNodes);


    	     double dueTime = getTimeOfService(_tempRoute, lastServedCustomer_id, serviceTime) + distance (dist,_tempRoute.back() , 0);


    	     double demandTour =  getCapacityUsed(_tempRoute);


    	     if (dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )
    	   	   	  {

    	       	  if ((!feasibleInsertion) || (costInsertion < lowCostInsertion))

    	   	   	     {

    	       		   lowCostInsertion= costInsertion;

    	   	   	       lowPosition = positionTour;

    	   	   	       _GenTempRoute = _tempRoute;

    	   	   	    	feasibleInsertion = true;

       	   	     }
       	   	  }
		  }
		}
	}else
	{

	    _tempRoute = _route ;

		positionTour = _tempRoute.size();

		_tempRoute.push_back(customer_id);

		costInsertion = distance(dist,_tempRoute.back(), customer_id) + distance(dist,customer_id, 0) - distance(dist,_tempRoute.back(),0) ;

		double dueTime = getTimeOfService(_tempRoute, lastServedCustomer_id, serviceTime) + distance (dist,_tempRoute.back() , 0);

	    double demandTour =  getCapacityUsed(_tempRoute) + clients[customer_id].demand;

	    if (dueTime <= depotTimeWindow && demandTour <= VEHICULE_CAPACITY )

	   	   {

	    	if ((!feasibleInsertion) || (costInsertion < lowCostInsertion))

	    		{

	    		 lowCostInsertion= costInsertion;

	        	 lowPosition = positionTour;

	    		 _GenTempRoute = _tempRoute;

	    		 feasibleInsertion = true;

	    		}
	   	   }

	  }


	if(feasibleInsertion)

		if( !commitCustomer || lowCostInsertion < generalizedTourCost )

		{
			generalizedTourCost  = lowCostInsertion;

			positionGeneralizedTour =  lowPosition;

			generalizedTour = indexTour;

			_Genroute = _GenTempRoute;

			commitCustomer = true;


		}
}

/*
double computTourLength(const  Routes _routes)
        {

       	 double Length = 0.0 ;


       	 for(size_t tr = 0, tsize = _routes.size(); tr < tsize; ++tr )

       	 {

       	  for (size_t i = 0, size = _routes[tr].size()-1; i < size; ++i )

             	   Length+= distance(dist,_routes[tr][i], _routes[tr][i+1]);



       	 	Length+=  distance(dist,_routes[tr].back(),0);

       	 }

      	 return Length ;

        }

*/


unsigned bestNeighborhoodRoutes ( const std::vector<Routes> _neighborhoodRoutes, unsigned _bestNeighborIndex, double _bestNeighbor_fitness)

{
	double minFitness, fitnessRoute ;

	unsigned minRoute ;


	minFitness = computTourLength(_neighborhoodRoutes[0]);

	minRoute  =  0;


	for (size_t i =1, size = _neighborhoodRoutes.size(); i < size ; i++ )

	{ fitnessRoute = computTourLength(_neighborhoodRoutes[i]) ;


		if(fitnessRoute < minFitness)


		{ minFitness = fitnessRoute;

		  minRoute =  i ;

		}

	  }

		_bestNeighborIndex = minRoute;

		_bestNeighbor_fitness = minFitness ;
}



void swapCustomers(Route &_route, unsigned _lastCustomerPosition)
{


	 unsigned customer1  = randPosition(_lastCustomerPosition + 1 , _route.size());


	 unsigned customer2 = randPosition(_lastCustomerPosition +1 , _route.size());


	swap(_route[customer1],_route[customer2]);

}




bool twoOptOnRoute( Route & _route , unsigned _lastServedCustomerPosition)

{
	bool twoOptFeasible = false ;

	for (unsigned from = _lastServedCustomerPosition , size1 = _route.size() ; from < size1 -1 ; from ++)

		for(unsigned to = from +1 , size2 = _route.size(); to < size2 ; to ++)

			if( GainCost(_route,from,to) > 0 )

				{
				  int idx =(to-from)/2;

				  for(unsigned k = 1; k <= idx ;k++)

				  std::swap(_route[from+k],_route[to-k]);



				   twoOptFeasible = true;

	            }

	return twoOptFeasible;
}



void InsertionTypeOne(Route & _route, unsigned positionGeneralizedTour, unsigned customer_id, std::vector <unsigned>  vertices)
{

	  _route.insert(_route.begin() + positionGeneralizedTour, customer_id );

	  unsigned j = vertices[0];

	  unsigned k = vertices[1];

	  _route.insert(_route.begin() + positionGeneralizedTour + 1, _route[j+1]);

	  _route.erase(_route.begin() + j + 2);

	  unsigned to = k + 1;

	  unsigned from =   positionGeneralizedTour + 2;

	  int idx =( to - from )/2;

	  for(unsigned q = 1; q <= idx ;q++)

		  std::swap(_route[from+q],_route[to-q]);


}

void threeMoveRoute(Route & _route, unsigned _from, unsigned _to, unsigned _end)
{

	Route tempRoute;

	for(size_t i=0; i<= _from; i++)

		tempRoute.push_back(_route[i]);

	tempRoute.push_back( _route[_to + 1]);

	tempRoute.push_back(_route[_end]);

	for(size_t i=_from +1; i<= _to; i++)

	tempRoute.push_back(_route[i]);

	tempRoute.push_back(_route[_end+1]);

	for(size_t i=_to +2; i<_end; i++)

		tempRoute.push_back(_route[i]);

	for(size_t i=_end + 2; i<_route.size(); i++)

		tempRoute.push_back(_route[i]);

	_route = tempRoute;


}

bool threeOptOnRoute( Route & _route , unsigned _lastServedCustomerPosition)

{
	bool threeOptFeasible = false ;

	for (unsigned from = _lastServedCustomerPosition , size1 = _route.size()-3 ; from < size1  ; from ++)

		for(unsigned to = from +1 , size2 = _route.size()-2; to < size2; to ++)

			for(unsigned end= to + 1, size3 = _route.size()-1 ; end < size3; end++)

			if( GainCost(_route,from,to,end) > 0 )

				{

                  //threeMoveRoute(_route,from,to,end);

				  if (!threeOptFeasible) threeOptFeasible = true;


	            }

	return threeOptFeasible;
}





double GainCost(Route _route, unsigned _from, unsigned _to)
{

	 double Gain ;

	 Gain= (distance (dist, _route[_from], _route[(_from +1) % _route.size()]) + distance(dist, _route[_to - 1], _route[_to % _route.size()]))

			 -

			 (distance (dist, _route[_from],_route[_to - 1]) + distance(dist, _route[(_from+1)%_route.size()],_route[_to% _route.size()]));

	 return Gain;
}


double GainCost(Route _route, unsigned _from, unsigned _to, unsigned _end)
{

	double Gain;

	Gain =  distance (dist, _route[_from],_route[(_from +1) % _route.size()]) +

			distance(dist, _route[_to], _route[(_to + 1) % _route.size()]) +

			distance (dist,_route[_end],_route[(_end +  1) % _route.size()])

			-

	        distance (dist, _route[_from],_route[(_to + 1) % _route.size()]) +

			distance(dist, _route[_to], _route[(_end + 1) % _route.size()]) +

			distance (dist,_route[_end],_route[(_from + 1) % _route.size()]);



	return Gain;

}



