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
#ifndef EOPSODVRPEVALFUNC_H_
#define EOPSODVRPEVALFUNC_H_

// General includes
#include <stdexcept>
#include <fstream>
#include "eoEvalFunc.h"

#include "eoPsoDVRP.h"





template <class POT>
class eoPsoDVRPEvalFunc : public eoEvalFunc<POT> {

public:

    /**
      * \brief Constructor: nothing to do here.
      */

	eoPsoDVRPEvalFunc () {}


    /**
      * \brief Computes the (penalized) fitness
      * \param _eo The individual to be evaluated.
      */

    void operator () (POT& _po) {

        if ( _po.invalid ())

        	{
        	      _po.toursLength(  computTourLength(_po.pRoutes));

        		  _po.fitness(  _po.toursLength());

            }


      if (_po.invalidBest ())

         {
    	  _po.bestToursLength( computTourLength(_po.bestRoutes));

          _po.best( _po.bestToursLength());

        }

        }

    void operator() (eoPop <POT> &_pop){

        	for (size_t i=0, size = _pop.size(); i < size; ++i)

        		operator()(_pop[i]);

        }

};



#endif /*EOPSODVRPEVALFUNC_H_*/
