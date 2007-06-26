// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoBitVector.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOBITVECTOR_H_
#define MOEOBITVECTOR_H_

#include <core/moeoVector.h>

/**
 * This class is an implementationeo of a simple bit-valued moeoVector.
 */
template < class MOEOObjectiveVector, class MOEOFitness, class MOEODiversity >
class moeoBitVector : public moeoVector < MOEOObjectiveVector, MOEOFitness, MOEODiversity, bool >
{
public:

    using moeoVector < MOEOObjectiveVector, MOEOFitness, MOEODiversity, bool > :: begin;
    using moeoVector < MOEOObjectiveVector, MOEOFitness, MOEODiversity, bool > :: end;
    using moeoVector < MOEOObjectiveVector, MOEOFitness, MOEODiversity, bool > :: resize;
    using moeoVector < MOEOObjectiveVector, MOEOFitness, MOEODiversity, bool > :: size;


    /**
     * Ctor
     * @param _size Length of vector (default is 0)
     * @param _value Initial value of all elements (default is default value of type GeneType)
     */
    moeoBitVector(unsigned int _size = 0, bool _value = false) : moeoVector< MOEOObjectiveVector, MOEOFitness, MOEODiversity, bool >(_size, _value)
    {}


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
            std::transform(bits.begin(), bits.end(), begin(), std::bind2nd(std::equal_to<char>(), '1'));
        }
    }

};

#endif /*MOEOBITVECTOR_H_*/
