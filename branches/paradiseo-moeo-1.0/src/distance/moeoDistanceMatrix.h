// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoDistanceMatrix.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEODISTANCEMATRIX_H_
#define MOEODISTANCEMATRIX_H_

#include <eoFunctor.h>
#include <distance/moeoDistance.h>

/**
 * A matrix to compute distances between every pair of individuals contained in a population. 
 */
template < class MOEOT , class Type >
class moeoDistanceMatrix : public eoUF < const eoPop < MOEOT > &, void > , public std::vector< std::vector < Type > >
{
public:

	using std::vector< std::vector < Type > > :: size;
    using std::vector< std::vector < Type > > :: operator[];
    
	
	/**
	 * Ctor
	 * @param _size size for every dimension of the matrix
	 * @param _distance the distance to use
	 */
	moeoDistanceMatrix (unsigned _size, moeoDistance < MOEOT , Type > & _distance) : distance(_distance)
	{
		this->resize(_size);
		for(unsigned i=0; i<_size; i++)
		{
			this->operator[](i).resize(_size);
		}
	}
	
	
	/**
	 * Sets the distance between every pair of individuals contained in the population _pop
	 * @param _pop the population
	 */
	void operator()(const eoPop < MOEOT > & _pop)
    {
    	// 1 - setup the bounds (if necessary)
    	distance.setup(_pop);
    	// 2 - compute distances
    	this->operator[](0).operator[](0) = Type();
    	for (unsigned i=0; i<size(); i++)
    	{
    		this->operator[](i).operator[](i) = Type();
    		for (unsigned j=0; j<i; j++)
    		{
    			this->operator[](i).operator[](j) = distance(_pop[i], _pop[j]);
    			this->operator[](j).operator[](i) = this->operator[](i).operator[](j);
    		}
    	}
    }


private:

	/** the distance to use */
	moeoDistance < MOEOT , Type > & distance;

};

#endif /*MOEODISTANCEMATRIX_H_*/
