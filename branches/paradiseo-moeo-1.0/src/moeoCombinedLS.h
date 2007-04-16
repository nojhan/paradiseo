// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoCombinedLS.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOCOMBINEDLS_H_
#define MOEOCOMBINEDLS_H_

#include <moeoArchive.h>
#include <moeoLS.h>

/**
 * This class allows to embed a set of local searches that are sequentially applied,
 * and so working and updating the same archive of non-dominated solutions.
 */
template < class MOEOT, class Type >
class moeoCombinedLS : public moeoLS < MOEOT, Type >
{
public:

    /**
     * Ctor
     * @param _first_mols the first multi-objective local search to add
     */
    moeoCombinedLS(moeoLS < MOEOT, Type > & _first_mols)
    {
        combinedLS.push_back (& _first_mols);
    }

    /**
     * Adds a new local search to combine
     * @param _mols the multi-objective local search to add
     */
    void add(moeoLS < MOEOT, Type > & _mols)
    {
        combinedLS.push_back(& _mols);
    }

    /**
     * Gives a new solution in order to explore the neigborhood. 
     * The new non-dominated solutions are added to the archive
     * @param _type the object to apply the local search to
     * @param _arch the archive of non-dominated solutions 
     */
    void operator () (Type _type, moeoArchive < MOEOT > & _arch)
    {
        for (unsigned i=0; i<combinedLS.size(); i++)
            combinedLS[i] -> operator()(_type, _arch);
    }


private:

    /** the vector that contains the combined LS */
    std::vector< moeoLS < MOEOT, Type > * >  combinedLS;

};

#endif /*MOEOCOMBINEDLS_H_*/
