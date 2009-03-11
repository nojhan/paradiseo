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
 * therefore means  that it is reserved for developers  and  exper10ienced
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
#ifndef EOPSODVRPENCODESWARM_H_
#define EOPSODVRPENCODESWARM_H_

#include"eoPsoDVRPutils.h"




template < class POT >

class eoPsoDVRPEncodeSwarm
{

public:

	eoPsoDVRPEncodeSwarm(){}


	void operator()(eoPop<POT> & _pop, double _tstep, double _tslice)

	{

/*
	vector<unsigned> temp = newCustomers;



	if (_insertionMethode == "Hybrid")
    {

		  int i =0 ;

		  int size = (int)(_pop.size()) / 3;


		while(i < size)
		{

			 if (_rebuild =="Rebuild")

				_pop[i].cleanParticle();

			_pop[i].encodingCurrentPositionRandomInsertion(_tstep, _tslice);

			  if(_rebuild == "Rebuild")

				newCustomers = temp;

			i++ ;

		}


		size = (int)(_pop.size()) *  (3/4);



		while (i < size)

		{

		if (_rebuild =="Rebuild")

			_pop[i].cleanParticle();

		 _pop[i].encodingCurrentPositionNearestInsertion(_tstep,  _tslice);

		 if(_rebuild == "Rebuild")

			newCustomers = temp;

		 i++ ;


		}


		size = (int)(_pop.size());


		while( i <  size)
		{
			if (_rebuild =="Rebuild")

				_pop[i].cleanParticle();

			_pop[i].encodingCurrentPositionCheapestInsertion(_tstep,  _tslice);

			 /*if(_rebuild == "Rebuild")

				newCustomers = temp;


			i++;

		}

	}

	else
	{
*/
		for (size_t i =0, size = _pop.size() ; i<size ; ++i)

			_pop[i].encodingCurrentPositionRandomInsertion(_tstep, _tslice);

			/*	{
			       if (_rebuild =="Rebuild")

			    	   _pop[i].cleanParticle();

		   		 if(_insertionMethode == "Rand")

		   			 _pop[i].encodingCurrentPositionRandomInsertion(_tstep, _tslice);

		   		 if(_insertionMethode == "Nearest")//

*/
		   			// _pop[i].encodingCurrentPositionNearestInsertion(_tstep,  _tslice);
/*
				  if(_insertionMethode == "Cheapest")*/

//				  _pop[i].encodingCurrentPositionCheapestInsertion(_tstep,  _tslice);

				/*  if(_rebuild == "Rebuild")

					  newCustomers = temp;
	             }


	}*/



	}


	void commitOrders(eoPop<POT> & _pop, double _tstep, double _tslice)

		{

			for (size_t i =0, size = _pop.size(); i<size ; ++i)

				  {

					_pop[i].commitOrdersCurrentPosition(_tstep, _tslice);


					_pop[i].commitOrdersBestPosition(_tstep, _tslice);

				  }

		}



	void initParticleToItsBest(eoPop<POT> & _pop){

		for (size_t i =0, size = _pop.size() ; i<size ; ++i)
		{
		_pop[i].copyBestToCurrent();

		for (size_t k = 0 , size = _pop[i].size(); k < size ; k++ )
			_pop[i].planifiedCustomers[k].pRouting = _pop[i].planifiedCustomers[k].bestRouting;

		_pop[i].pRoutes  = _pop[i].bestRoutes;
	    _pop[i].pLength = _pop[i].bestLength;
	    _pop[i].firstTimeServiceCurrentPosition = _pop[i].firstTimeServiceBestPosition;
		_pop[i].fitness(_pop[i].bestLength);

		}
	}



};



#endif /*EOPSODVRPENCODESWARM_H_*/
