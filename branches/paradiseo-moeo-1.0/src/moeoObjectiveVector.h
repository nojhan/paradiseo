// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoObjectiveVector.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOOBJECTIVEVECTOR_H_
#define MOEOOBJECTIVEVECTOR_H_

#include <iostream>
#include <math.h>
#include <vector>
#include <moeoObjectiveVectorComparator.h>

/**
 * Abstract class allowing to represent a solution in the objective space (phenotypic representation). 
 * The template argument ObjectiveVectorTraits defaults to moeoObjectiveVectorTraits, 
 * but it can be replaced at will by any other class that implements the static functions defined therein.
 * Some static funtions to access to the traits characteristics are re-defined in order not to write a lot of typedef's.
 */
template < class ObjectiveVectorTraits >
class moeoObjectiveVector
{
public:

	/** The traits of objective vectors */
	typedef ObjectiveVectorTraits Traits;
	
	
	/**
	 * Parameters setting (for the objective vector of any solution)
	 * @param _nObjectives the number of objectives
	 * @param _bObjectives the min/max vector (true = min / false = max)
	 */
	static void setup(unsigned _nObjectives, std::vector < bool > & _bObjectives)
	{
		ObjectiveVectorTraits::setup(_nObjectives, _bObjectives);
	}
	
	
	/**
	 * Returns the number of objectives
	 */
	static unsigned nObjectives()
	{
		return ObjectiveVectorTraits::nObjectives();
	}
	
	
	/**
	 * Returns true if the _ith objective have to be minimized
	 * @param _i  the index
	 */
	static bool minimizing(unsigned _i) {
		return ObjectiveVectorTraits::minimizing(_i);
	}
	
	
	/**
	 * Returns true if the _ith objective have to be maximized
	 * @param _i  the index
	 */
	static bool maximizing(unsigned _i) {
		return ObjectiveVectorTraits::maximizing(_i);
	}
		
};


/**
 * This class allows to represent a solution in the objective space (phenotypic representation) by a std::vector of doubles,
 * i.e. that an objective value is represented using a double, and this for any objective. 
 */
template < class ObjectiveVectorTraits >
class moeoObjectiveVectorDouble : public moeoObjectiveVector < ObjectiveVectorTraits >, public std::vector < double >
{
public:	
	
	using std::vector< double >::size;	
	using std::vector< double >::operator[];
		
	/**
	 * Ctor
	 */
	moeoObjectiveVectorDouble() : std::vector < double > (ObjectiveVectorTraits::nObjectives(), 0.0) {}	
		
	
	/**
	 * Ctor from a vector of doubles
	 * @param _v the std::vector < double >
	 */
	moeoObjectiveVectorDouble(std::vector <double> & _v) : std::vector < double > (_v) {}
	
	
	/**
	 * Returns true if the current objective vector dominates _other according to the Pareto dominance relation
	 * (but it's better to use a moeoObjectiveVectorComparator object to compare solutions)
	 * @param _other the other moeoObjectiveVectorDouble object to compare with
	 */
	bool dominates(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
	{
		moeoParetoObjectiveVectorComparator < moeoObjectiveVectorDouble<ObjectiveVectorTraits> > comparator;
		return comparator(*this, _other);
	}
	
	
	/**
	 * Returns true if the current objective vector is equal to _other (according to a tolerance value)
	 * @param _other the other moeoObjectiveVectorDouble object to compare with
	 */
	bool operator==(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
	{
		for (unsigned i=0; i < size(); i++)
		{
			if ( fabs(operator[](i) - _other[i]) > ObjectiveVectorTraits::tolerance() )
			{
				return false;
			}      
		}
		return true;
	}
	
	
	/**
	 * Returns true if the current objective vector is different than _other (according to a tolerance value)
	 * @param _other the other moeoObjectiveVectorDouble object to compare with 
	 */
	bool operator!=(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
	{
		return ! operator==(_other);
	}
	
	
	/**
	 * Returns true if the current objective vector is smaller than _other on the first objective, then on the second, and so on
	 * (can be usefull for sorting/printing)
	 * @param _other the other moeoObjectiveVectorDouble object to compare with
	 */ 
	bool operator<(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
	{
		for (unsigned i=0; i < size(); i++)
		{
			if ( fabs(operator[](i) - _other[i]) > ObjectiveVectorTraits::tolerance() )
			{
				if (operator[](i) < _other[i])
				{
					return true;
				}
				else
				{
					return false;
				}
			}
		}
		return false;
	}
	
	
	/**
	 * Returns true if the current objective vector is greater than _other on the first objective, then on the second, and so on
	 * (can be usefull for sorting/printing)
	 * @param _other the other moeoObjectiveVectorDouble object to compare with
	 */
	bool operator>(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
	{
		return _other < *this;
	}
	
	
	/**
	 * Returns true if the current objective vector is smaller than or equal to _other on the first objective, then on the second, and so on
	 * (can be usefull for sorting/printing)
	 * @param _other the other moeoObjectiveVectorDouble object to compare with
	 */
	bool operator<=(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
	{
		return operator==(_other) || operator<(_other);
	}
	
	
	/**
	 * Returns true if the current objective vector is greater than or equal to _other on the first objective, then on the second, and so on
	 * (can be usefull for sorting/printing)
	 * @param _other the other moeoObjectiveVectorDouble object to compare with
	 */
	bool operator>=(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
	{
		return operator==(_other) || operator>(_other);
	}	

};


/**
 * Output for a moeoObjectiveVectorDouble object
 * @param _os output stream
 * @param _objectiveVector the objective vector to write
 */
template < class ObjectiveVectorTraits >
std::ostream & operator<<(std::ostream & _os, const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _objectiveVector)
{
	for (unsigned i=0; i<_objectiveVector.size(); i++)
	{
		_os << _objectiveVector[i] << ' ';
	} 
	return _os;
}

/**
 * Input for a moeoObjectiveVectorDouble object
 * @param _is input stream
 * @param _objectiveVector the objective vector to read
 */
template < class ObjectiveVectorTraits >
std::istream & operator>>(std::istream & _is, moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _objectiveVector)
{
	_objectiveVector = moeoObjectiveVectorDouble < ObjectiveVectorTraits > ();
	for (unsigned i=0; i<_objectiveVector.size(); i++)
	{
		_is >> _objectiveVector[i];
	} 
	return _is;
}

#endif /*MOEOOBJECTIVEVECTOR_H_*/
