// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoHybridLS.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOHYBRIDLS_H_
#define MOEOHYBRIDLS_H_

#include <eoContinue.h>
#include <eoPop.h>
#include <eoSelect.h>
#include <utils/eoUpdater.h>
#include <algo/moeoLS.h>
#include <archive/moeoArchive.h>

/**
 * This class allows to apply a multi-objective local search to a number of selected individuals contained in the archive
 * at every generation until a stopping criteria is verified.
 */
template < class MOEOT >
class moeoHybridLS : public eoUpdater
{
public:

    /**
     * Ctor
     * @param _term stopping criteria
     * @param _select selector
     * @param _mols a multi-objective local search
     * @param _arch the archive
     */
    moeoHybridLS (eoContinue < MOEOT > & _term, eoSelect < MOEOT > & _select, moeoLS < MOEOT, MOEOT > & _mols, moeoArchive < MOEOT > & _arch) :
            term(_term), select(_select), mols(_mols), arch(_arch)
    {}


    /**
     * Applies the multi-objective local search to selected individuals contained in the archive if the stopping criteria is not verified 
     */
    void operator () ()
    {
        if (! term (arch))
        {
            // selection of solutions
            eoPop < MOEOT > selectedSolutions;
            select(arch, selectedSolutions);
            // apply the local search to every selected solution
            for (unsigned int i=0; i<selectedSolutions.size(); i++)
            {
                mols(selectedSolutions[i], arch);
            }
        }
    }


private:

    /** stopping criteria */
    eoContinue < MOEOT > & term;
    /** selector */
    eoSelect < MOEOT > & select;
    /** multi-objective local search */
    moeoLS < MOEOT, MOEOT > & mols;
    /** archive */
    moeoArchive < MOEOT > & arch;

};

#endif /*MOEOHYBRIDLS_H_*/
