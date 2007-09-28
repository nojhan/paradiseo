// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoVector.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
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
template < class MOEOObjectiveVector, class MOEOFitness, class MOEODiversity, class GeneType >
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
            	throw std::runtime_error("Size not initialized in moeoVector");
            }
        }
        std::copy(_v.begin(), _v.end(), begin());
        invalidate();
    }


    /**
     * To avoid conflicts between MOEO::operator< and std::vector<GeneType>::operator<
     * @param _moeo the object to compare with
     */
    bool operator<(const moeoVector< MOEOObjectiveVector, MOEOFitness, MOEODiversity, GeneType> & _moeo) const
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
bool operator<(const moeoVector< MOEOObjectiveVector, MOEOFitness, MOEODiversity, GeneType> & _moeo1, const moeoVector< MOEOObjectiveVector, MOEOFitness, MOEODiversity, GeneType> & _moeo2)
{
    return _moeo1.operator<(_moeo2);
}


/**
 * To avoid conflicts between MOEO::operator> and std::vector<double>::operator>
 * @param _moeo1 the first object to compare
 * @param _moeo2 the second object to compare
 */
template < class MOEOObjectiveVector, class MOEOFitness, class MOEODiversity, class GeneType >
bool operator>(const moeoVector< MOEOObjectiveVector, MOEOFitness, MOEODiversity, GeneType> & _moeo1, const moeoVector< MOEOObjectiveVector, MOEOFitness, MOEODiversity, GeneType> & _moeo2)
{
    return _moeo1.operator>(_moeo2);
}

#endif /*MOEOVECTOR_H_*/
