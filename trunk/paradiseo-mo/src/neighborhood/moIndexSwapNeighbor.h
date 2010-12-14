/*
 <moIndexSwapNeighbor.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Boufaras Karima, Thé Van Luong

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  ue,
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

#ifndef _moIndexSwapNeighbor_h
#define _moIndexSwapNeighbor_h

#include <neighborhood/moBackableNeighbor.h>
#include <neighborhood/moIndexNeighbor.h>

/**
 * Index Swap Neighbor
 */

template<class EOT, class Fitness = typename EOT::Fitness>
class moIndexSwapNeighbor: public moIndexNeighbor<EOT> {
public:

	using moIndexNeighbor<EOT>::key;

	/**
	 * Default Constructor
	 */

	moIndexSwapNeighbor() :
		moIndexNeighbor<EOT> () {
		Kswap = 0;
		indices = new unsigned int[Kswap + 1];
	}

/*	moIndexSwapNeighbor(unsigned int _Kswap) :
		moIndexNeighbor<EOT> () {
		Kswap = _Kswap;
		indices = new unsigned int[Kswap + 1];
	}*/

	/**
	 * Default destructor
	 */

	~moIndexSwapNeighbor() {
		delete[] (indices);
	}

	/**
	 * Constructor
	 * @param _Kswap the number of swap to do
	 */

	moIndexSwapNeighbor(unsigned int _Kswap) {
		Kswap = _Kswap;
		indices = new unsigned int[Kswap + 1];
	}

	/**
	 * Copy Constructor
	 * @param _n the neighbor to copy
	 */

	moIndexSwapNeighbor(const moIndexSwapNeighbor& _n) :
		moIndexNeighbor<EOT> (_n) {
		this->Kswap = _n.Kswap;
		this->indices = new unsigned int[Kswap + 1];
		;
		for (unsigned int i = 0; i <= Kswap; i++)
			this->indices[i] = _n.indices[i];
	}

	/**
	 * Assignment operator
	 * @param _source the source neighbor
	 */

	virtual moIndexSwapNeighbor<EOT> & operator=(
			const moIndexSwapNeighbor<EOT> & _source) {
		moIndexNeighbor<EOT, Fitness>::operator=(_source);
		this->key = _source.key;
		this->Kswap = _source.Kswap;
		this->reSizeIndices(Kswap);
		for (unsigned int i = 0; i <= Kswap; i++)
			this->indices[i] = _source.indices[i];
		return *this;
	}

	/**
	 * Setter to update the i'th index of k-swap
	 * @param _i the index to update
	 * @param _val the new value the i'th index of k-swap
	 */

	void setIndice(unsigned int _i, unsigned int _val) {
		if (_i <= Kswap)
			if (_val < size) {
				indices[_i] = _val;
			} else
				std::cout << "The element " << _val
						<< " is out of range value " << std::endl;
		else
			std::cout << "The element " << _i << " is out of bound "
					<< std::endl;
	}

	/**
	 * Get the i'th index of k-swap
	 */

	unsigned int getIndice(unsigned int _i) {
		return indices[_i];
	}

	/**
	 * Setter to update the set of k-swap indexes
	 * @param _indices the set of new value indexes of the k-swap
	 */

	void setIndices(unsigned int * _indices) {
		for (unsigned int i = 0; i <= Kswap; i++)
			setIndice(i, _indices[i]);
	}

	/**
	 * Get the set of k-swap indexes
	 */

	unsigned int * getIndices() {
		return indices;
	}

	/**
	 * Setter to fix the Kswap
	 * @param _Kswap the number of swap
	 */

	void setKswap(unsigned int _Kswap) {
		Kswap = _Kswap;
	}

	/**
	 * Get the number of swap
	 */

	unsigned int getKswap() {
		return Kswap;
	}

	/**
	 * Setter to fix the neighbor size
	 * @param _size the neighbor size to set
	 */

	void setSize(unsigned int _size) {
		size = _size;
	}

	/**
	 * Get the size of neighbor
	 */

	unsigned int getSize() {
		return size;
	}

	/**
	 * Resize the indices array of K-swap indexes
	 *@param _Kswap the number of swap
	 */

	void reSizeIndices(unsigned int _Kswap) {
		delete[] (indices);
		indices = new unsigned int[_Kswap + 1];
	}

	/**
	 * Move a solution
	 * @param _solution the related solution
	 */
	virtual void move(EOT & _solution) {
	}
	;

	/**
	 * Move Back a solution
	 * @param _solution the related solution
	 */
	virtual void moveBack(EOT & _solution) {
	}
	;

	/**
	 * Return the class name.
	 * @return the class name as a std::string
	 */

	virtual std::string className() const {
		return "moIndexSwapNeighbor";
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
			_is >> size;
			_is >> Kswap;
			_is >> key;
			for (unsigned int i = 0; i <= Kswap; i++)
				_is >> indices[i];
			fitness(repFit);
		}
	}

	/**
	 * Print the Neighbor
	 */

	void print() {
		std::cout << "[";
		for (int i = 0; i <= Kswap; i++)
			std::cout << " " << indices[i];
		std::cout << "] -> " << (*this).fitness() << std::endl;
	}

protected:

	unsigned int * indices;
	unsigned int Kswap;
	unsigned int size;

};

#endif

