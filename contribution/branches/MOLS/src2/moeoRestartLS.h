/*
* <moeoRestartLS.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jérémie Humeau
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

#ifndef _MOEORESTARTLS_H
#define _MOEORESTARTLS_H

#include <eo>
#include <moeo>
#include <moeoPopLS.h>
#include <moeoUnifiedDominanceBasedLS.h>
#include <moeoNewArchive.h>
#include <moeoPopNeighborhoodExplorer.h>
#include <moeoUnvisitedSelect.h>

template < class Move >
class moeoRestartLS : public moeoPopLS < Move >
{
public:

    typedef typename Move::EOType MOEOT;

    moeoRestartLS(
        eoInit < MOEOT > & _init,
        eoEvalFunc < MOEOT > & _eval,
        eoContinue < MOEOT > & _continuator,
        moeoPopNeighborhoodExplorer < Move > & _explorer,
        moeoUnvisitedSelect < Move > & _select,
        moeoArchive < MOEOT > & _globalArchive,
        std::string _fileName) :
            init(_init), eval(_eval), continuator(_continuator), ls(continuator, _eval, internalArchive, _explorer, _select), globalArchive(_globalArchive), fileName(_fileName), count(0) {}


    virtual void operator()(eoPop<MOEOT> & _pop)
    {
        do
        {
            internalArchive.resize(0);
            for (unsigned int i=0; i<_pop.size(); i++)
            {
                init(_pop[i]);
                _pop[i].invalidateObjectiveVector();
                eval(_pop[i]);
            }
            ls(_pop);
            count++;
            globalArchive(internalArchive);
        } while (continuator(globalArchive));
        save();
//         std::cout << "Final archive\n";
//         globalArchive.sortedPrintOn(std::cout);
//         std::cout << std::endl;
    }


protected:

    eoInit < MOEOT > & init;
    eoEvalFunc < MOEOT > & eval;
    eoContinue < MOEOT > & continuator;
    moeoNewArchive < MOEOT > internalArchive;
    moeoUnifiedDominanceBasedLS < Move > ls;
    moeoArchive < MOEOT > & globalArchive;
    std::string & fileName;
    unsigned int count;


    void save()
    {
        // save count in a file
        std::string tmp = fileName;
        tmp += ".stat";
        std::ofstream outfile(tmp.c_str());
        outfile << count << std::endl;
        outfile.close();
    }

};

#endif /*_MOEORESTARTLS_H*/
