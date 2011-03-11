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


#ifndef EOPSODVRPINIT_H_
#define EOPSODVRPINIT_H_

#include <eo>
#include "eoPsoDVRPEvalFunc.h"
#include "eoDVRPStarTopology.h"
#include "eoPsoDVRPutils.h"


template <class POT>

class eoNothingInit: public  eoInit<POT>
{

public:

	eoNothingInit(){}

	virtual void operator()(POT& _po) {

		_po.resize(0);

		_po.invalidate ();


	}
};



class eoInitialSettings
{
	public:


	eoInitialSettings( string _benchmarkFile, const double TIME_CUT_OFF, const unsigned NBR_SLICES): benchmarkFile(_benchmarkFile)

	{TIME_CUTOFF = TIME_CUT_OFF;

	 NBRSLICES = NBR_SLICES;

	 operator()();

    }

	void operator() () {


		loadMatrixCustomers(benchmarkFile);

		computeDistances();

		SetFleetCapacity(benchmarkFile);

		FindTimeDay(benchmarkFile);

		TIME_STEP = TIME_ADVANCE = 0.0;

		TIME_SLICE = TIME_DAY/NBRSLICES;


}

	void SetFleetCapacity(string filename){
			   istringstream  iss;
			   string line;
			   ifstream filein (filename.c_str());
			   IsReadable(filein);
			   do

				getline(filein,line);

				while(line.find("NUM_VEHICLES")== string::npos);
				iss.str(line);
				iss>>line>>FLEET_VEHICLES;
				iss.clear();
				do
					getline(filein,line);

				while(line.find("CAPACITIES")== string::npos);

				iss.str(line);

				iss>>line>> VEHICULE_CAPACITY;

				iss.clear();

				filein.close();

		}

	void FindTimeDay(string filename){

			ifstream file (filename.c_str());

			string line;

			unsigned int customer;

			double time;

			istringstream  iss;

			do

				getline(file,line);


			while(line.find("DEPOT_TIME_WINDOW_SECTION")== string::npos);


			getline(file,line);

			iss.str(line);

			iss>>customer>>customer>>TIME_DAY;

			iss.clear();

			file.close();

		}

	private:

		string benchmarkFile;



};


/*

template < class POT >

class eoParticleInit : public eoInit<POT>
{

public :


	eoParticleInit(){}

	void operator()(POT & _po) {


		//_po. eoVectorParticle<eoMinimizingFitness,int,int>::operator = _po.bestPositions;

		for (size_t i = 0 , size = _po.size(); i < size ; i++ )

			_po.planifiedCustomers[i].pRouting = _po.planifiedCustomers[i].bestRouting;

		_po.pRoutes  = _po.bestRoutes;

		_po.pLength = _po.bestLength;

		_po.firstTimeServiceCurrentPosition = _po.firstTimeServiceBestPosition;

		_po.fitness(_po.bestLength);

	}



	void operator () (eoPop<POT> &_pop) {

		for (size_t i = 0, size = _pop.size(); i < size ; i++)
		{

			operator()(_pop[i]);

		}


	}



};

*/


template < class POT >

class  eoDVRBestPositionInit: public  eoParticleBestInit <POT>  // First initializer of swarm
{

public:

	eoDVRBestPositionInit () {}

	void operator  () (POT & _po)
	 {

		 _po.bestPositions = _po ;

		  _po.bestLength = _po.pLength ;

		 _po.bestRoutes =_po.pRoutes;

		 //il faut une �valuation avant sinon le invalidate best sera faux

		 _po.best(_po.fitness());

		 if ( !std::equal( _po.bestPositions.begin(), _po.bestPositions.end(), _po.begin() ) )

			 std::cerr << "The copy was not performed correctly when initializeBestPosition"<<endl;


	      for (size_t i=0 , size = _po.planifiedCustomers.size(); i < size ; ++i)

	      {  _po.planifiedCustomers[i].bestRouting.route = _po.planifiedCustomers[i].pRouting.route ;

	         _po.planifiedCustomers[i].bestRouting.routePosition = _po.planifiedCustomers[i].pRouting.routePosition ;

	         _po.planifiedCustomers[i].bestRouting.is_served= _po.planifiedCustomers[i].pRouting.is_served;

	         _po.planifiedCustomers[i].bestRouting.serviceTime= _po.planifiedCustomers[i].pRouting.serviceTime;


	      }


	      _po.firstTimeServiceBestPosition = _po.firstTimeServiceCurrentPosition ;

	 }
};




#endif /*EOPSODVRPINIT_H_*/
