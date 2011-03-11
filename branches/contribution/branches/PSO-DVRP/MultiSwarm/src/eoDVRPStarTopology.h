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

#ifndef EODVRPSTARTOPOLOGY_H_
#define EODVRPSTARTOPOLOGY_H_

#include <eo>

template < class POT >

class eoDVRPStarTopology:public eoTopology <POT>
 {
 public:


eoDVRPStarTopology(): eoTopology<POT>(), globalBest(){}



void setup(const eoPop<POT> & _pop)
{


	globalBest = bestNeighbour(_pop);


}



void updateNeighborhood(POT & _po,unsigned _indice)
{

	if( _po.fitness() < _po.best())
	{

		 _po.best(_po.pLength);

		 _po.bestRoutes = _po.pRoutes;

		 _po.bestLength = _po.pLength ;


		 for(size_t i =0, size = _po.planifiedCustomers.size(); i < size ; ++i)

			 _po.planifiedCustomers[i].bestRouting = _po.planifiedCustomers[i].pRouting;



		 _po.firstTimeServiceBestPosition = _po.firstTimeServiceCurrentPosition;


	}


	if ( _po.best() < globalBest.best())

		globalBest = _po ;

}




	POT bestNeighbour (const eoPop<POT>& _pop)

	{
		 unsigned index_best =0 ;

		 if(_pop.size() == 0)

			 cerr<<"You want to get the bestNeighbour from empty swarm...!!!!!!"<<endl;

		 for (size_t i=1, size = _pop.size(); i < size ; ++i)

			 if (_pop[i].best() < _pop[index_best].best())  //tester et v�rifier

				  index_best = i;


		 return _pop[index_best];

	}


	virtual POT &  best (unsigned i = 0){return globalBest; } ;


	private:

		POT   globalBest;



};





#endif /*EODVRPSTARTOPOLOGY_H_*/
