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

#ifndef _eoPsoDVRP_h
#define _eoPsoDVRP_h


#include "eoPsoDVRPutils.h"



using eo::rng;



typedef struct record1 {

					int  route;

	                int  routePosition;

	                bool is_served;

	                double serviceTime;

	 					} routingInfo;



typedef struct record2 {

						unsigned id;

						routingInfo pRouting;

						routingInfo bestRouting;

						int velocity;

						}particleRouting;




//using namespace eoParticuleDVRPUtils; eoMinimizingFitness

class eoParticleDVRP: public  eoVectorParticle <double,int,int> {

public:

	eoParticleDVRP();

	virtual ~eoParticleDVRP ();

	eoParticleDVRP(const eoParticleDVRP& _orig);

	void copy(const eoParticleDVRP & _po);

	eoParticleDVRP& operator= (const eoParticleDVRP& _orig);

    virtual std::string className () const ;

    double toursLength ();

    void toursLength (const double _pLength);

    double bestToursLength();

    void bestToursLength (const double _bestLength);

    bool clean ();

    bool cleanCurrentRoutes ();

    bool cleanBestRoutes ();

    void printRoutesOn(std::ostream& _os) const;

    void printBestRoutesOn(std::ostream& _os) const;

    void printOn(std::ostream& _os) const;

    void printBestOn(std::ostream& _os) const;

    void printVelocities(std::ostream& _os);

    void printfirstTimeService(std::ostream& _os);

    void printfirstBestTimeService(std::ostream& _os);

    void printCurrentPosition(std::ostream& _os)const;

    void printBestPosition(std::ostream& _os)const;

    void computToursLength();

    void computCurrentTourLength();

    void computBestTourLength();

    void setVelocities();

    void setCurrentPositions ();

    void setBestPositions();

    void encodingCurrentPositionCheapestInsertion(double _tstep, double _tslice);

    void encodingCurrentPositionNearestInsertion(double _tstep, double _tslice);

    void encodingCurrentPositionGeneralizedInsertion(double _tstep, double _tslice);

    void encodingCurrentPositionRandomInsertion(double _tstep, double _tslice);

    void encodingBestPosition(double _tstep, double _tslice);

    routingInfo & currentRoutingCustomer(unsigned id_customer);

    routingInfo & bestRoutingCustomer(unsigned id_customer);

    void commitOrdersCurrentPosition (double _nextTstep, double _timeSlice);

    void commitOrdersBestPosition (double _nextTstep, double _timeSlice);

    bool isServedCustomerInCurrentPosition (unsigned id_customer);

    bool isServedCustomerInBestPosition (unsigned id_customer);

    void normalizeVelocities();

    void eraseRoute(unsigned _tour);

    void checkCurrentPosition();

    void checkBestPosition();

    bool  serviceIsProgressCurrentPosition(unsigned _tour);

    bool  serviceIsProgressBestPosition(unsigned _tour);

    unsigned IndexLastServedCustomerCurrentPosition(unsigned _tour);

    unsigned IndexLastServedCustomerBestPosition(unsigned _tour);

    double  timeEndServiceCurrentPosition(unsigned id_customer);

    double  timeEndServiceBestPosition(unsigned id_customer);

    void cleanParticle();

    void copyBestToCurrent();

    void lastPositionOnRoutes(std::vector<unsigned> & _positions);

    void reDesign ();

    void checkAll(std::ostream& _os);



//private:

   std::vector<particleRouting> planifiedCustomers;

   double pLength , bestLength;

   Routes pRoutes,  bestRoutes;

   std::vector<double> firstTimeServiceCurrentPosition;

   std::vector<double> firstTimeServiceBestPosition;


};


#endif


