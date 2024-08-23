/*
* <moeoBitVector.h>
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

#ifndef MOEOBITVECTOR_H_
#define MOEOBITVECTOR_H_

#include <core/moeoVector.h>

/**
 * This class is an implementationeo of a simple bit-valued moeoVector.
 */
template < class MOEOObjectiveVector, class MOEOFitness=double, class MOEODiversity=double >
class moeoBitVector : public moeoVector < MOEOObjectiveVector, bool, MOEOFitness, MOEODiversity >
  {
  public:

    using moeoVector < MOEOObjectiveVector, bool, MOEOFitness, MOEODiversity > :: begin;
    using moeoVector < MOEOObjectiveVector, bool, MOEOFitness, MOEODiversity > :: end;
    using moeoVector < MOEOObjectiveVector, bool, MOEOFitness, MOEODiversity > :: resize;
    using moeoVector < MOEOObjectiveVector, bool, MOEOFitness, MOEODiversity > :: size;


    /**
     * Ctor
     * @param _size Length of vector (default is 0)
     * @param _value Initial value of all elements (default is default value of type GeneType)
     */
    moeoBitVector(unsigned int _size = 0, bool _value = false) : moeoVector< MOEOObjectiveVector, bool, MOEOFitness, MOEODiversity >(_size, _value)
    {}


    /**
     * Returns the class name as a std::string
     */
    virtual std::string className() const
      {
        return "moeoBitVector";
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
        std::copy(begin(), end(), std::ostream_iterator<bool>(_os));
      }


    /**
    * Reading object
    * @param _is input stream
    */
    virtual void readFrom(std::istream & _is)
    {
      MOEO < MOEOObjectiveVector, MOEOFitness, MOEODiversity >::readFrom(_is);
      unsigned int s;
      _is >> s;
      std::string bits;
      _is >> bits;
      if (_is)
        {
          resize(bits.size());
#if __cplusplus >= 201103L
          std::transform(bits.begin(), bits.end(), begin(), std::bind(std::equal_to<char>(), std::placeholders::_1, '1'));
#else
          std::transform(bits.begin(), bits.end(), begin(), std::bind2nd(std::equal_to<char>(), '1'));
#endif
        }
    }

  };

#endif /*MOEOBITVECTOR_H_*/
