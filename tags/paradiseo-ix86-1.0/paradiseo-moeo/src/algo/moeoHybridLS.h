/* <moeoHybridLS.h>  
 *
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
 */
 
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
