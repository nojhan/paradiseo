/*
* <MOEO.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
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

#ifndef MOEO_H_
#define MOEO_H_

#include <iostream>
#include <stdexcept>
#include <string>
#include "../../eo/EO.h"

/**
 * Base class allowing to represent a solution (an individual) for multi-objective optimization.
 * The template argument MOEOObjectiveVector allows to represent the solution in the objective space (it can be a moeoObjectiveVector object).
 * The template argument MOEOFitness is an object reflecting the quality of the solution in term of convergence (the fitness of a solution is always to be maximized).
 * The template argument MOEODiversity is an object reflecting the quality of the solution in term of diversity (the diversity of a solution is always to be maximized).
 * All template arguments must have a void and a copy constructor.
 * Using some specific representations, you will have to define a copy constructor if the default one is not what you want.
 * In the same cases, you will also have to define the affectation operator (operator=).
 * Then, you will explicitly have to call the parent copy constructor and the parent affectation operator at the beginning of the corresponding implementation.
 * Besides, note that, contrary to the mono-objective case (and to EO) where the fitness value of a solution is confused with its objective value,
 * the fitness value differs of the objectives values in the multi-objective case.
 */

/*
 template < typename DataType, typename DataTypeEx > struct Wrapper {

	Wrapper() {}

	Wrapper( const DataType& data )
		: embededData( data ) {}

	Wrapper( const Wrapper& wrapper )
		: embededData( wrapper.embededData ), embededDataEx( wrapper.embededDataEx ) {}

	operator const DataType& () const { return embededData; }

	Wrapper& operator= ( const Wrapper& wrapper ) {

		embededData = wrapper.embededData;
		embededDataEx = wrapper.embededDataEx;
		return *this;
	}

	Wrapper& operator= ( const DataType& data ) {

		embededData = data;
		return *this;
	}

	DataType embededData;
	DataTypeEx embededDataEx;
};
 **/


template < class MOEOObjectiveVector, class MOEOFitness=double, class MOEODiversity=double >
class MOEO : public EO < MOEOObjectiveVector >
  {
  public:

    /** the objective vector type of a solution */
    typedef MOEOObjectiveVector ObjectiveVector;

    /** the fitness type of a solution */
    typedef MOEOFitness Fitness;
//	typedef Wrapper< MOEOFitness, MOEOObjectiveVector > Fitness;

    /** the diversity type of a solution */
    typedef MOEODiversity Diversity;


    /**
     * Ctor
     */
    MOEO()
    {
      // default values for every parameters
      objectiveVectorValue = ObjectiveVector();
      fitnessValue = Fitness();
      diversityValue = Diversity();
      // invalidate all
      invalidate();
      flagValue=0;
    }


    /**
     * Virtual dtor
     */
    virtual ~MOEO()
    {};


    /**
     * Returns the objective vector of the current solution
     */
    ObjectiveVector objectiveVector() const
      {
        if ( invalidObjectiveVector() )
          {
            throw std::runtime_error("invalid objective vector in MOEO");
          }
        return objectiveVectorValue;
      }


    /**
     * Sets one dimension of the objective vector 
     * @param _dim dimension of the objective vector to set
     * @param _value the new value of the corresponding objective
     */
  void objectiveVector(unsigned int _dim, typename ObjectiveVector::Type _value)
    {
      objectiveVectorValue[_dim] = _value;
      invalidObjectiveVectorValue = false;
    }


    /**
     * Gets one dimension of the objective vector 
     * @param _dim dimension of the objective vector to set
     */
  typename ObjectiveVector::Type objectiveVector(unsigned int _dim) const
    {
      return objectiveVectorValue[_dim];
    }


    /**
     * Sets the objective vector of the current solution
     * @param _objectiveVectorValue the new objective vector
     */
    void objectiveVector(const ObjectiveVector & _objectiveVectorValue)
    {
      objectiveVectorValue = _objectiveVectorValue;
      invalidObjectiveVectorValue = false;
    }


    /**
     * Sets the objective vector as invalid
     */
    void invalidateObjectiveVector()
    {
      invalidObjectiveVectorValue = true;
    }


    /**
     * Returns true if the objective vector is invalid, false otherwise
     */
    bool invalidObjectiveVector() const
      {
        return invalidObjectiveVectorValue;
      }


    /**
     * Returns the fitness value of the current solution
     */
    Fitness fitness() const
      {
        if ( invalidFitness() )
          {
            throw std::runtime_error("invalid fitness in MOEO");
          }
//         const_cast< Fitness& >( fitnessValue ).embededDataEx = objectiveVectorValue;

        return fitnessValue;
      }


    /**
     * Sets the fitness value of the current solution
     * @param _fitnessValue the new fitness value
     */
    void fitness(const Fitness & _fitnessValue)
    {
      fitnessValue = _fitnessValue;
      invalidFitnessValue = false;
    }


    /**
     * Sets the fitness value as invalid
     */
    void invalidateFitness()
    {
      invalidFitnessValue = true;
    }


    /**
     * Returns true if the fitness value is invalid, false otherwise
     */
    bool invalidFitness() const
      {
        return invalidFitnessValue;
      }


    /**
     * Returns the diversity value of the current solution
     */
    Diversity diversity() const
      {
        if ( invalidDiversity() )
          {
            throw std::runtime_error("invalid diversity in MOEO");
          }
        return diversityValue;
      }


    /**
     * Sets the diversity value of the current solution
     * @param _diversityValue the new diversity value
     */
    void diversity(const Diversity & _diversityValue)
    {
      diversityValue = _diversityValue;
      invalidDiversityValue = false;
    }


    /**
     * Sets the diversity value as invalid
     */
    void invalidateDiversity()
    {
      invalidDiversityValue = true;
    }


    /**
     * Returns true if the diversity value is invalid, false otherwise
     */
    bool invalidDiversity() const
      {
        return invalidDiversityValue;
      }


    /**
     * Sets the objective vector, the fitness value and the diversity value as invalid
     */
    void invalidate()
    {
      invalidateObjectiveVector();
      invalidateFitness();
      invalidateDiversity();
    }


    /**
     * Returns true if the objective values are invalid, false otherwise
     */
    bool invalid() const
      {
        return invalidObjectiveVector();
      }


    /**
     * Returns true if the objective vector of the current solution is smaller than the objective vector of _other on the first objective,
     * then on the second, and so on (can be usefull for sorting/printing).
     * You should implement another function in the sub-class of MOEO to have another sorting mecanism.
     * @param _other the other MOEO object to compare with
     */
    bool operator<(const MOEO & _other) const
      {
        return objectiveVector() < _other.objectiveVector();
      }


    /**
     * Return the class id (the class name as a std::string)
     */
    virtual std::string className() const
      {
        return "MOEO";
      }


    /**
     * Writing object
     * @param _os output stream
     */
    virtual void printOn(std::ostream & _os) const
      {
        if ( invalidObjectiveVector() )
          {
            _os << "INVALID ";
          }
        else
          {
            _os << objectiveVectorValue << ' ';
          }
      }


    /**
     * Reading object
     * @param _is input stream
     */
    virtual void readFrom(std::istream & _is)
    {
      std::string objectiveVector_str;
      int pos = _is.tellg();
      _is >> objectiveVector_str;
      if (objectiveVector_str == "INVALID")
        {
          invalidateObjectiveVector();
        }
      else
        {
          invalidObjectiveVectorValue = false;
          _is.seekg(pos); // rewind
          _is >> objectiveVectorValue;
        }
    }

    /**
     * Setter for "flag"
     * @param _flag the flag value
     */
    void flag(int _flag){
    	flagValue=_flag;
    }

    /**
     * Getter for "flag"
     * @return the flag value
     */
    int flag() const{
    	return flagValue;
    }


  private:

    /** the objective vector of this solution */
    ObjectiveVector objectiveVectorValue;
    /** true if the objective vector is invalid */
    bool invalidObjectiveVectorValue;
    /** the fitness value of this solution */
    Fitness fitnessValue;
    /** true if the fitness value is invalid */
    bool invalidFitnessValue;
    /** the diversity value of this solution */
    Diversity diversityValue;
    /** true if the diversity value is invalid */
    bool invalidDiversityValue;
    /** A flag which can be used to stock information*/
    int flagValue;

  };



#endif /*MOEO_H_*/
