/*
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * (c) Antonio LaTorre <atorre@fi.upm.es>, 2007
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

#ifndef eoVRPUtils_h
#define eoVRPUtils_h

// General includes
#include <vector>
#include <utility>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>

/**
  * \def PI
  * Guess you know what this constant represents.
  */

#define PI                   3.14159265

/**
  * \def VEHICLE_CAPACITY
  * Hard-coded parameter for the capacity of the vehicles. This
  * should be parametrized in a config file in a future version.
  */

#define VEHICLE_CAPACITY   200


typedef std::vector<int> Route;
typedef std::vector< Route > Routes;


/**
  * \namespace eoVRPUtils
  * \brief A set of structures and utility functions for the VRP-TW problem.
  */

namespace eoVRPUtils {

/**
* \var typedef struct ClientData ClientDataT.
* \brief Renaming of struct ClientData.
*/

/**
* \struct ClientData
* \brief Information regarding each client in the dataset.
* This structure is intended to be used to store the information of each
* client read from the data file.
*/

typedef struct ClientData {

    unsigned id;            /**< Client ID number. */
    double   x;             /**< Client's 'x' position in the map. */
    double   y;             /**< Client's 'y' position in the map. */
    double   demand;        /**< Client's demand of delivered product. */
    double   readyTime;     /**< Client's beginning of the time window. */
    double   dueTime;       /**< Client's end of the time window. */
    double   serviceTime;   /**< Client's service time (time needed to serve the product). */

} ClientDataT;


static std::vector <ClientDataT> clients;             /**< Vector to store clients's information. */
static std::vector <std::vector <double> > dist;      /**< Distance matrix. */


/**
   * \brief Computes the distance between two clients.
   * The computed distances will be stored in dist.
   */

void computeDistances () {

    unsigned numClients = clients.size ();

    dist.resize (numClients) ;

    for (unsigned i = 0; i < dist.size (); i ++)
        dist [i].resize (numClients);

    // Distances computation
    for (unsigned i = 0; i < dist.size (); i ++)
        for (unsigned j = i + 1 ; j < dist.size (); j ++) {

            double distX = clients [i].x - clients [j].x;
            double distY = clients [i].y - clients [j].y;

            dist [i][j] = dist [j][i] = sqrt (distX * distX + distY * distY);

        }

}


/**
   * \brief Returns the time window information of a given client.
   * \param _client The client whose information we want to know.
   * \param _readyTime Return value. The beginning of the client's time window.
   * \param _dueTime Return value. The end of the client's time window.
   * \param _serviceTime Return value. The client's service time.
   */

void getTimeWindow (unsigned _client, double& _readyTime, double& _dueTime, double& _serviceTime) {

    assert (_client >= 0 && _client < clients.size ());

    _readyTime = clients [_client].readyTime;
    _dueTime = clients [_client].dueTime;
    _serviceTime = clients [_client].serviceTime;

}


/**
   * \brief A function to get the distance between two clients.
   * \param _from The first client.
   * \param _to The second client.
   * \return The distance between _from and _to.
   */

float distance (unsigned _from,  unsigned _to) {

    assert (_from >= 0 && _from < clients.size ());
    assert (_to   >= 0 && _to   < clients.size ());

    return dist [_from][_to];

}


/**
   * \brief Computes de polar angle between clients.
   * \param _from The first client.
   * \param _to The second client.
   * \return The polar angle between _from and _to.
   */

float polarAngle (unsigned _from, unsigned _to) {

    assert (_from >= 0 && _from < clients.size ());
    assert (_to   >= 0 && _to   < clients.size ());

    double angle = atan2 (clients [_from].y - clients [_to].y,
                          clients [_from].x - clients [_to].x);

    // To convert it from radians to degrees
    angle *= 180 / PI;

    if (angle < 0)
        angle *= -1;

    return angle;

}


/**
   * \brief Loads the problem data from a given file.
   * \param _fileName The file to load data from.
   * \warning No error check is performed!
   */

void load (const char* _fileName) {

    std::ifstream f (_fileName);

    if (f) {

        while (!f.eof ()) {

            ClientDataT client;

            f >> client.id;
            f >> client.x;
            f >> client.y;
            f >> client.demand;
            f >> client.readyTime;
            f >> client.dueTime;
            f >> client.serviceTime;

            clients.push_back (client);

        }

        f.close ();

        computeDistances ();

    }
    else {

        std::cerr << "Error: the file: " << _fileName << " doesn't exist !!!" << std::endl ;
        exit (1);

    }

}


/**
  * \brief Prints a route to the standard output.
  * \param _route The route to print.
  */

void printRoute (const Route& _route) {

    std::cout << "[";

    for (unsigned i = 0; i < _route.size (); i++) {

        std::cout << _route [i];

        if (i != _route.size () -1)
            std::cout << ", ";

    }

    std::cout << "]";

}


/**
  * \brief Prints a set of routes to the standard output.
  * \param _routes The set of routes to print.
  */

void printRoutes (Routes& _routes) {

    std::cout << "[";

    for (unsigned i = 0; i < _routes.size (); i++) {

        std::cout << "[";

        for (unsigned j = 0; j < _routes [i].size (); j++) {

            std::cout << _routes [i][j];

            if (j != _routes [i].size () -1)
                std::cout << ", ";

        }

        if (i == _routes.size () -1)
            std::cout << "]";
        else
            std::cout << "]," << std::endl;
    }

    std::cout << "]";

}


};

#endif
