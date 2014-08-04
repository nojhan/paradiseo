/*
* <moeoNumberUnvisitedSelect.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jérémie Humeau
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
//-----------------------------------------------------------------------------

#ifndef _MOEONUMBERUNVISITEDSELECT_H
#define _MOEONUMBERUNVISITEDSELECT_H

#include "../../eo/eoPop.h"
#include "moeoUnvisitedSelect.h"

/**
 * Selector which select a part of unvisited individuals of a population
 */
template < class MOEOT >
class moeoNumberUnvisitedSelect : public moeoUnvisitedSelect < MOEOT >
{

public:

	/**
	 * Constructor
	 * @param _number the number of individuals to select
	 */
    moeoNumberUnvisitedSelect(unsigned int _number): number(_number){}

    /**
     * functor which return index of selected individuals of a population
     * @param _src the population
     * @return the vector contains index of the part of unvisited individuals of the population
     */
    std::vector <unsigned int> operator()(eoPop < MOEOT > & _src)
    {
    	std::vector <unsigned int> res;
    	res.resize(0);
        for (unsigned int i=0; i<_src.size(); i++)
        {
            if (_src[i].flag() == 0)
            	res.push_back(i);
        }
        if(number < res.size()){
        	UF_random_generator<unsigned int> rndGen;
        	std::random_shuffle(res.begin(), res.end(), rndGen);
        	res.resize(number);
        }
        return res;
    }

private:
	/** number of individuals to select */
	unsigned int number;

};

#endif /*_MOEONUMBERUNVISITEDSELECT_H_*/
