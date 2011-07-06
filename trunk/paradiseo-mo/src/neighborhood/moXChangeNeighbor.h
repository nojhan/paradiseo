/*
 <moXChangeNeighbor.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Boufaras Karima, Thé Van Luong

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

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

#ifndef _moXChangeNeighbor_h
#define _moXChangeNeighbor_h

#include <neighborhood/moIndexNeighbor.h>

/**
 * A X-Change Neighbor
 * useful in parallel computing, it allows to associate to a neighbor represented by a single index,a set of indices
 * to build the neighbor from the solution and the number of positions to change
 */

template<class EOT, class Fitness = typename EOT::Fitness>
class moXChangeNeighbor: public moIndexNeighbor<EOT> {

public:

	using moIndexNeighbor<EOT>::key;

	/**
	 * Default Constructor
	 */

	moXChangeNeighbor() :
		moIndexNeighbor<EOT> (),xChange(0) {
		indices = NULL;
	}

	/**
	 * Default destructor
	 */

	~moXChangeNeighbor() {
		delete[] (indices);
	}

	/**
	 * Constructor
	 * @param _xChange the number of x-change to do
	 */

	moXChangeNeighbor(unsigned int _xChange) {
		xChange = _xChange;
		indices = new unsigned int[xChange];
	}

	/**
	 * Copy Constructor
	 * @param _n the neighbor to copy
	 */

	moXChangeNeighbor(const moXChangeNeighbor& _n) :
		moIndexNeighbor<EOT> (_n) {
		this->xChange = _n.xChange;
		this->indices = new unsigned int[xChange];
		for (unsigned int i = 0; i < xChange; i++)
			this->indices[i] = _n.indices[i];
	}

	/**
	 * Assignment operator
	 * @param _source the neighbor to assign to this curent neighbor
	 */

	virtual moXChangeNeighbor<EOT> & operator=(
			const moXChangeNeighbor<EOT> & _source) {
		moIndexNeighbor<EOT, Fitness>::operator=(_source);
		this->xChange = _source.xChange;
		this->reSizeIndices(xChange);
		for (unsigned int i = 0; i < xChange; i++)
			this->indices[i] = _source.indices[i];
		return *this;
	}

	/**
	 * Setter to update the i'th index of x-change
	 * @param _i the index to update
	 * @param _val the new value to set to the i'th index of x-change
	 */

	void setIndice(unsigned int _i, unsigned int _val) {
			indices[_i] = _val;
	}

	/**
	 * Get the i'th index of x-change
	 */

	unsigned int getIndice(unsigned int _i) {
		return indices[_i];
	}

	/**
	 * Setter to update the set of x-change indexes
	 * @param _indices the set of new value indexes of the x-change
	 */

	void setIndices(unsigned int * _indices) {
		for (unsigned int i = 0; i < xChange; i++) {
			setIndice(i, _indices[i]);
		}
	}

	/**
	 * Setter to fix the xChange
	 * @param _xChange the number of swap
	 */

	void setXChange(unsigned int _xChange) {
		xChange = _xChange;
		reSizeIndices(_xChange);
	}

	/**
	 * Get the number of swap
	 */

	unsigned int getXChange() {
		return xChange;
	}

	/**
	 * Resize the indices array of x-change indexes
	 *@param _xChange the number of x-change
	 */

	void reSizeIndices(unsigned int _xChange) {
		delete[] (indices);
		indices = new unsigned int[_xChange];
	}

	/**
	 * Return the class name.
	 * @return the class name as a std::string
	 */

	virtual std::string className() const {
		return "moXChangeNeighbor";
	}

	/**
	 * Read object.\
	 * Calls base class, just in case that one had something to do.
	 * The read and print methods should be compatible and have the same format.
	 * In principle, format is "plain": they just print a number
	 * @param _is a std::istream.
	 * @throw runtime_std::exception If a valid object can't be read.
	 */

	virtual void readFrom(std::istream& _is) {
		std::string fitness_str;
		int pos = _is.tellg();
		_is >> fitness_str;
		if (fitness_str == "INVALID") {
			throw std::runtime_error("invalid fitness");
		} else {
			Fitness repFit;
			_is.seekg(pos);
			_is >> repFit;
			_is >> xChange;
			_is >> key;
			for (unsigned int i = 0; i < xChange; i++)
				_is >> indices[i];
			fitness(repFit);
		}
	}

	/**
	 * Print the Neighbor
	 */

	void print() {
		std::cout << "[";
		for (int i = 0; i < xChange; i++)
			std::cout<< indices[i]<< " ";
		std::cout << "] -> " << (*this).fitness() << std::endl;
	}

protected:

	unsigned int * indices;
	unsigned int xChange;

};

#endif

