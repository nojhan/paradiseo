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

#ifndef EOPSODVRPVELOCITY_H_
#define EOPSODVRPVELOCITY_H_

#include"eoPsoDVRP.h"


template < class POT >

class eoPsoDVRPvelocity:public eoVelocity < POT >
{

public:

	typedef typename POT::ParticleVelocityType VelocityType;

	eoPsoDVRPvelocity( eoDVRPStarTopology<POT> & _topology,

			 			const double & _w,

			 			const double & _c1,

			 			const double & _c2): topology(_topology),

			 			omega(_w),c1(_c1),c2(_c2){}



	 void updateNeighborhood(POT & _po,unsigned _indice)
	 {

		 topology.updateNeighborhood(_po,_indice);

	 }

	 void updateNeighborhood(eoPop<POT> & _pop,unsigned _indice = 0)
		 {

			 for (size_t i =0, size = _pop.size() ; i < size; i++)

				 topology.updateNeighborhood(_pop[i],_indice);

		 }


	 eoTopology<POT> & getTopology ()
	 {
		  return topology;
	 }




	 void operator()(POT & _po,unsigned _indice)  {


		 VelocityType newVelocity;


		 if(_po.size()!=_po.velocities.size() || _po.bestPositions.size()!= _po.velocities.size())

			 std::cerr<<"The size of the particle is different to the velocity size...!"<<endl;


		 if (_po.pRoutes.size() == 0 || _po.bestRoutes.size() ==0)

			 std::cerr<<" The current position tours, or the best position tours are empty ....!!!!"<<endl;


		 int Vlimit = _po.pRoutes.size()-1;


		 for (unsigned j = 0, size = _po.velocities.size();  j < size ; j++)
		 {

		   newVelocity= static_cast<int> (omega * _po.velocities[j] + c1 * (_po.bestPositions[j] - _po[j]) +  c2 * (topology.best(_indice).bestPositions[j] - _po[j]));


		   _po.velocities[j]= _po.planifiedCustomers[j].velocity =  randVelocity(_po.pRoutes.size()-1) ;//boundVelocity(newVelocity, Vlimit);


	     }

		 _po.normalizeVelocities();


   }


		/* eoRealVectorBounds bounds(_po.size(),-Vlimit,Vlimit);

		 bounds.adjust_size(_po.size());*/

	 /* if (bounds.isMinBounded(j))
	   newVelocity=std::max(newVelocity,bounds.minimum(j));
	   if (bounds.isMaxBounded(j))
		   newVelocity=std::min(newVelocity,bounds.maximum(j));*/



	 VelocityType boundVelocity(VelocityType velocity, int limit)
	 {
		 if(limit == 0)

			 return 0;
	     else

		 if(velocity < -limit)

		 		return (velocity % -limit);

		  else

			  if (velocity > limit)

		        return (velocity % limit);

			  else

				return velocity;
	 }


private:

	eoDVRPStarTopology<POT> & topology;

	const double & omega;  // social/cognitive coefficient

	const double & c1;  // social/cognitive coefficient

	const double & c2;  // social/cognitive coefficient


};





#endif /*EOPSODVRPVELOCITY_H_*/
