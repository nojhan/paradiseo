// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoObjectiveVectorTraits.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOOBJECTIVEVECTORTRAITS_H_
#define MOEOOBJECTIVEVECTORTRAITS_H_

#include <vector>
#include <iostream>
#include <stdexcept>

/**
 * A traits class for moeoObjectiveVector to specify the number of objectives and which ones have to be minimized or maximized.
 */
class moeoObjectiveVectorTraits
{
public:

    /**
     * Parameters setting
     * @param _nObjectives the number of objectives
     * @param _bObjectives the min/max vector (true = min / false = max)
     */
    static void setup(unsigned _nObjectives, std::vector < bool > & _bObjectives)
    {
        // in case the number of objectives was already set to a different value
        if ( nObj && (nObj != _nObjectives) ) {
            std::cout << "WARNING\n";
            std::cout << "WARNING : the number of objectives are changing\n";
            std::cout << "WARNING : Make sure all existing objects are destroyed\n";
            std::cout << "WARNING\n";
        }
        // number of objectives
        nObj = _nObjectives;
        // min/max vector
        bObj = _bObjectives;
        // in case the number of objectives and the min/max vector size don't match
        if (nObj != bObj.size())
            throw std::runtime_error("Number of objectives and min/max size don't match in moeoObjectiveVectorTraits::setup");
    }

    /**
     * Returns the number of objectives
     */
    static unsigned nObjectives()
    {
        // in case the number of objectives would not be assigned yet
        if (! nObj)
            throw std::runtime_error("Number of objectives not assigned in moeoObjectiveVectorTraits");
        return nObj;
    }

    /**
     * Returns true if the _ith objective have to be minimized
     * @param _i  the index
     */
    static bool minimizing(unsigned _i)
    {
        // in case there would be a wrong index
        if (_i >= bObj.size())
            throw std::runtime_error("Wrong index in moeoObjectiveVectorTraits");
        return bObj[_i];
    }

    /**
     * Returns true if the _ith objective have to be maximized
     * @param _i  the index
     */
    static bool maximizing(unsigned _i) {
        return (! minimizing(_i));
    }

    /**
     * Returns the tolerance value (to compare solutions)
     */
    static double tolerance()
    {
        return 1e-6;
    }


private:

    /** The number of objectives */
    static unsigned nObj;
    /** The min/max vector */
    static std::vector < bool > bObj;

};


// The static variables of the moeoObjectiveVectorTraits class need to be allocated
unsigned moeoObjectiveVectorTraits::nObj;
std::vector < bool > moeoObjectiveVectorTraits::bObj;


#endif /*MOEOOBJECTIVEVECTORTRAITS_H_*/
