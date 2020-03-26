/*
* <moeoVector.h>
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

#ifndef MOEOVECTOR_H_
#define MOEOVECTOR_H_

#include <iterator>
#include <vector>
#include <core/MOEO.h>

/**
 * Base class for fixed length chromosomes, just derives from MOEO and std::vector and redirects the smaller than operator to MOEO (objective vector based comparison).
 * GeneType must have the following methods: void ctor (needed for the std::vector<>), copy ctor.
 */
template < class MOEOObjectiveVector, class GeneType, class MOEOFitness=double, class MOEODiversity=double >
class moeoVector : public MOEO < MOEOObjectiveVector, MOEOFitness, MOEODiversity >, public std::vector < GeneType >
  {
  public:

    using MOEO < MOEOObjectiveVector, MOEOFitness, MOEODiversity > :: invalidate;
    using std::vector < GeneType > :: operator[];
    using std::vector < GeneType > :: begin;
    using std::vector < GeneType > :: end;
    using std::vector < GeneType > :: resize;
    using std::vector < GeneType > :: size;

    /** the atomic type */
    typedef GeneType AtomType;
    /** the container type */
    typedef std::vector < GeneType > ContainerType;


    /**
     * Default ctor.
     * @param _size Length of vector (default is 0)
     * @param _value Initial value of all elements (default is default value of type GeneType)
     */
    moeoVector(unsigned int _size = 0, GeneType _value = GeneType()) :
        MOEO < MOEOObjectiveVector, MOEOFitness, MOEODiversity >(), std::vector<GeneType>(_size, _value)
    {}


    /**
     * We can't have a Ctor from a std::vector as it would create ambiguity  with the copy Ctor.
     * @param _v a vector of GeneType
     */
    void value(const std::vector < GeneType > & _v)
    {
      if (_v.size() != size())	   // safety check
        {
          if (size())		   // NOT an initial empty std::vector
            {
              std::cout << "Warning: Changing size in moeoVector assignation"<<std::endl;
              resize(_v.size());
            }
          else
            {
              throw eoException("Size not initialized in moeoVector");
            }
        }
      std::copy(_v.begin(), _v.end(), begin());
      invalidate();
    }


    /**
     * To avoid conflicts between MOEO::operator< and std::vector<GeneType>::operator<
     * @param _moeo the object to compare with
     */
    bool operator<(const moeoVector< MOEOObjectiveVector, GeneType, MOEOFitness, MOEODiversity> & _moeo) const
      {
        return MOEO < MOEOObjectiveVector, MOEOFitness, MOEODiversity >::operator<(_moeo);
      }


    /**
     * Writing object
     * @param _os output stream
     */
    virtual void printOn(std::ostream & _os) const
      {
        MOEO < MOEOObjectiveVector, MOEOFitness, MOEODiversity >::printOn(_os);
        _os << ' ';
        _os << size() << ' ';
        std::copy(begin(), end(), std::ostream_iterator<AtomType>(_os, " "));
      }


    /**
     * Reading object
     * @param _is input stream
     */
    virtual void readFrom(std::istream & _is)
    {
      MOEO < MOEOObjectiveVector, MOEOFitness, MOEODiversity >::readFrom(_is);
      unsigned int sz;
      _is >> sz;
      resize(sz);
      unsigned int i;
      for (i = 0; i < sz; ++i)
        {
          AtomType atom;
          _is >> atom;
          operator[](i) = atom;
        }
    }

  };


/**
 * To avoid conflicts between MOEO::operator< and std::vector<double>::operator<
 * @param _moeo1 the first object to compare
 * @param _moeo2 the second object to compare
 */
template < class MOEOObjectiveVector, class MOEOFitness, class MOEODiversity, class GeneType >
bool operator<(const moeoVector< MOEOObjectiveVector, GeneType, MOEOFitness, MOEODiversity> & _moeo1, const moeoVector< MOEOObjectiveVector, GeneType, MOEOFitness, MOEODiversity > & _moeo2)
{
  return _moeo1.operator<(_moeo2);
}


/**
 * To avoid conflicts between MOEO::operator> and std::vector<double>::operator>
 * @param _moeo1 the first object to compare
 * @param _moeo2 the second object to compare
 */
template < class MOEOObjectiveVector, class MOEOFitness, class MOEODiversity, class GeneType >
bool operator>(const moeoVector< MOEOObjectiveVector, GeneType, MOEOFitness, MOEODiversity> & _moeo1, const moeoVector< MOEOObjectiveVector, GeneType, MOEOFitness, MOEODiversity > & _moeo2)
{
  return _moeo1.operator>(_moeo2);
}

#endif /*MOEOVECTOR_H_*/
