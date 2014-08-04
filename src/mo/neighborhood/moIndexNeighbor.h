/*
 <moIndexNeighbor.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef _IndexNeighbor_h
#define _IndexNeighbor_h

#include "moNeighbor.h"

/**
 * Index Neighbor
 */
template<class EOT, class Fitness = typename EOT::Fitness>
class moIndexNeighbor: virtual public moNeighbor<EOT, Fitness> {
public:

	using moNeighbor<EOT, Fitness>::fitness;

	/**
	 * Default Constructor
	 */
	moIndexNeighbor() :
		moNeighbor<EOT, Fitness> (), key(0) {
	}

	/**
	 * Copy Constructor
	 * @param _n the neighbor to copy
	 */
	moIndexNeighbor(const moIndexNeighbor& _n) :
		moNeighbor<EOT, Fitness> (_n) {
		this->key = _n.key;
	}

	/**
	 * Assignment operator
	 * @param _source the source neighbor
	 */

	moIndexNeighbor<EOT, Fitness> & operator=(const moIndexNeighbor<EOT,
			Fitness> & _source) {
		moNeighbor<EOT, Fitness>::operator=(_source);
		this->key = _source.key;
		return *this;
	}

	/**
	 * Return the class Name
	 * @return the class name as a std::string
	 */
	virtual std::string className() const {
		return "moIndexNeighbor";
	}

	/**
	 * Getter
	 * @return index of the IndexNeighbor
	 */
	inline unsigned int index() const {
		return key;
	}

	/**
	 * Setter : 
	 * Only set the index which not depends on the current solution
	 *
	 * @param _key index of the IndexNeighbor
	 */
  	void index(unsigned int _key) {
	  key = _key;
	}
  
  
	/**
	 * Setter 
	 * The "parameters" of the neighbor is a function of key and the current solution
	 * for example, for variable length solution
	 *
	 * @param _solution solution from which the neighborhood is visited
	 * @param _key index of the IndexNeighbor
	 */
  virtual void index(EOT & _solution, unsigned int _key) {
    key = _key;
  }

	/**
	 * @param _neighbor a neighbor
	 * @return if _neighbor and this one are equals
	 */
	virtual bool equals(moIndexNeighbor<EOT>& _neighbor) {
		return (key == _neighbor.index());
	}

protected:
	// key allowing to describe the neighbor
	unsigned int key;

};

#endif
