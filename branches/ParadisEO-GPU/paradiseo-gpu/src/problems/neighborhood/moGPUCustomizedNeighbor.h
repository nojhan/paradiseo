/*
 <moGPUCustomizedNeighbor.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Jerémie Humeau, Boufaras Karima, Thé Van LUONG

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

 As a counterpart to the access to the source code and  rights to copy,
 modify and redistribute granted by the license, users are provided only
 with a limited warranty  and the software's author,  the holder of the
 economic rights,  and the successive licensors  have only  limited liability.

 In this respect, the user's attention is drawn to the risks associated
 with loading,  using,  modifying and/or developing or reproducing the
 software by the user in light of its specific status of free software,
 that may mean  that it is complicated to manipulate,  and  that  also
 therefore means  that it is reserved for developers  and  experienced
 professionals having in-depth computer knowledge. Users are therefore
 encouraged to load and test the software's suitability as regards their
 requirements in conditions enabling the security of their systems and/or
 data to be ensured and,  more generally, to use and operate it in the
 same conditions as regards security.
 The fact that you are presently reading this means that you have had
 knowledge of the CeCILL license and that you accept its terms.

 ParadisEO WebSite : http://paradiseo.gforge.inria.fr
 Contact: paradiseo-help@lists.gforge.inria.fr
 */

#ifndef _moGPUCustomizedNeighbor_h
#define _moGPUCustomizedNeighbor_h

#include <neighborhood/moBackableNeighbor.h>
#include <neighborhood/moGPUXChangeNeighbor.h>
#include <problems/types/moGPUSolType2Vector.h>

/**
 * Neighbor related to a solution vector composed by two vectors
 */

template<class Fitness>
class moGPUCustomizedNeighbor: public moBackableNeighbor< moGPUSolType2Vector<Fitness> > ,
public moGPUXChangeNeighbor< moGPUSolType2Vector<Fitness> > {

public:

	using moGPUXChangeNeighbor< moGPUSolType2Vector<Fitness> >::indices;
	using moGPUXChangeNeighbor< moGPUSolType2Vector<Fitness> >::xChange;
	using moGPUXChangeNeighbor< moGPUSolType2Vector<Fitness> >::key;
	/**
	 *Default Constructor
	 */

	moGPUCustomizedNeighbor() :
		moGPUXChangeNeighbor< moGPUSolType2Vector<Fitness> > () {
	}

	/**
	 * Constructor
	 * @param _xSwap the number of bit to swap
	 */

	moGPUCustomizedNeighbor(unsigned int _xSwap) :
		moGPUXChangeNeighbor< moGPUSolType2Vector<Fitness> > (_xSwap) {
	}

	/**
	 * move the solution
	 * @param _solution the solution to move
	 */

	virtual void move(moGPUSolType2Vector<Fitness> & _solution) {
		std::cout<<"_solution"<<std::endl;
		float tmp;
		tmp = _solution[0].tab2[indices[0]];
		_solution[0].tab2[indices[0]] = _solution[0].tab2[indices[1]];
		_solution[0].tab2[indices[1]] = tmp;
		std::cout<<_solution<<std::endl;
		_solution.invalidate();
	}

	/**
	 * apply the moveBack to restore the solution (use by moFullEvalByModif)
	 * @param _solution the solution to move back
	 */

	virtual void moveBack(moGPUSolType2Vector<Fitness> & _solution) {
		move(_solution);
	}

	/**
	 * Return the class name.
	 * @return the class name as a std::string
	 */
	virtual std::string className() const {
		return "moGPUCustomizedNeighbor";
	}

};

#endif
