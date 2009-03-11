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

#ifndef eoPsoDVRPUTILS_H_
#define eoPsoDVRPUTILS_H_

// General includes
#include <vector>
#include <utility>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <eo>
#include "eoEventScheduler.h"

#include "eoGlobal.h"


using namespace std;


 typedef std::vector<unsigned> Route;

 typedef std::vector< Route > Routes;


 typedef struct record {

     unsigned id;   		  /**< Client ID number. */

     double   availTime;      /**<Client's time available.*/

     double   x;             /**< Client's 'x' position in the map. */

     double   y;             /**< Client's 'y' position in the map. */

     double   demand;        /**< Client's demand of delivered product. */

     double   durationService;   /**< Client's service time (time needed to serve the product). */

 }  ClientDataT;


 typedef std::vector <ClientDataT> setCustumers;

 extern setCustumers clients;  // The matrix relative to the customers

 extern std::vector<unsigned> newCustomers;

 typedef  std::vector <std::vector <double> > lengthMatrix ;      /**< Distance matrix. */

 extern lengthMatrix dist; // The matrix of distances between the customers*/

 template<typename T> std::string to_string( const T & Value );

 void computeDistances ();   // Distance calculation

 double computTourLength(Route _route);

 double computTourLength(Routes _mRoute);

void getInformation (unsigned _custumer, double& _availTime, double& _durationService); // Obtain certain information about the customer

 double distance  (const lengthMatrix  _dist, unsigned _from, unsigned  _to); // Calculate the distance between two customers

 bool findClient (unsigned id); // Test if the customer exists

 void loadMatrixCustomers (const string _fileName); // Load the benchmark file and fill the customers matrix

 void loadNewCustomers(double _timeStep, double _timeSlice);

 void PrintCustumers(); //Print the Customers matrix and also  distance matrix

 void PrintLastCustomers(); // Print the Customers who are comming in the last time slice

 void printRoute (const Route& _route);

 void printRoutes (const Routes& _routes, std::ostream &_os);

 const  setCustumers SetKnowClients();  // Return the Customers matrix

 const lengthMatrix  distanceMatrix();  //Return the distance matrix

 double getTimeOfService (Route _route, double _startTime, unsigned _newCustomer_id, unsigned _newPosition);

 double getTimeOfService (Route _route, unsigned _lastServedCustomer, double _serviceTime, unsigned _newCustomer_id, unsigned _newPosition);

 double getTimeOfService(const Route _route, unsigned _lastServedCustomer_id, double dispo);

 double startServiceTime (double _nextTimeStep, double _dispoTime);

 double getCapacityUsed (Route _route);

 Route emptyRoute();

 unsigned randTour(unsigned maxTour);

 int randVelocity (int _sizeTour);

 unsigned randPosition (unsigned _lastPosition, unsigned _sizeTour);

 void checkDistances(const Routes _matrix);

 void checkCharge(const Routes _matrix);

 void visualization (const Routes _matrix , unsigned _seed, ofstream & _os);

 double computTourLength(Routes _routes);

void cheapestInsertionAlgorithm (Route _route, unsigned indexTour,

								unsigned lastServedCustomer_id, unsigned lastServedCustomerPosition,

								double serviceTime, unsigned customer_id, unsigned & cheapestTour,

								double & cheapestTourCost, unsigned & positionCheapestTour, bool & commitCustomer);

void cheapestInsertionAlgorithm (Route _route, unsigned indexTour,

								double _startTime, unsigned customer_id, unsigned & cheapestTour,double & cheapestTourCost,

								unsigned & positionCheapestTour, bool & commitCustomer) ;


void nearestInsertionAlgorithm (Route _route, unsigned indexTour,

								 unsigned lastServedCustomer_id, unsigned lastServedCustomerPosition,

								 double serviceTime, unsigned customer_id, unsigned & nearestTour,

								 double & nearestTourCost, unsigned & positionNearestTour, bool & commitCustomer);

void nearestInsertionAlgorithm (Route _route, unsigned indexTour,

								double _startTime, unsigned customer_id, unsigned & nearestTour,double & nearestTourCost,

								unsigned & positionNearestTour, bool & commitCustomer);


void generalizedInsertionTypeOneAlgorithm (Route _route, Route& _Genroute, unsigned indexTour,  unsigned lastServedCustomer_id,

										  unsigned lastServedCustomerPosition, double serviceTime, unsigned customer_id,

										  unsigned & generalizedTour, unsigned &  positionGeneralizedTour, double & generalizedTourCost, bool & commitCustomer);



void generalizedInsertionTypeOneAlgorithm (Route _route, Route & _GenRoute,	unsigned indexTour, double _startTime, unsigned customer_id, unsigned & generalizedTour,

										unsigned & positionGeneralizedTour, double & generalizedTourCost, bool & commitCustomer);


void InsertionTypeOne(Route& _route, unsigned positionGeneralizedTour, unsigned customer_id, std::vector<unsigned> vertices);

void swapCustomers(Route & _route, unsigned _lastCustomerPosition);

bool twoOptOnRoute( Route & _route , unsigned from);

double GainCost(Route _route, unsigned _from, unsigned _to);

double GainCost(Route _route, unsigned _from, unsigned _to, unsigned _end);

void threeMoveRoute(Route & _route, unsigned _from, unsigned _to, unsigned _end);

bool threeOptOnRoute( Route & _route , unsigned _lastServedCustomerPosition);
#endif /*eoPsoDVRPUTILS_H_*/
