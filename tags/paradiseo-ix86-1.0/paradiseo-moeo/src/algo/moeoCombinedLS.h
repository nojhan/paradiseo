/* <moeoCombinedLS.h>  
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
 
#ifndef MOEOCOMBINEDLS_H_
#define MOEOCOMBINEDLS_H_

#include <vector>
#include <algo/moeoLS.h>
#include <archive/moeoArchive.h>

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
        for (unsigned int i=0; i<combinedLS.size(); i++)
            combinedLS[i] -> operator()(_type, _arch);
    }


private:

    /** the vector that contains the combined LS */
    std::vector< moeoLS < MOEOT, Type > * >  combinedLS;

};

#endif /*MOEOCOMBINEDLS_H_*/
