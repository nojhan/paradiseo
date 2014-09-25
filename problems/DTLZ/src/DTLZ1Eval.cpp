/*
* <DTLZ1Eval.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jeremie Humeau
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

#include <DTLZ1Eval.h>

#define M_PI 3.14159265358979323846
// Ce code est implémenté à partir de DEME:Dtlz1.cpp
void DTLZ1Eval::operator() (DTLZ & _element)
{
    if (_element.invalidObjectiveVector())
    {
        int nbFun= DTLZ::ObjectiveVector::nObjectives();
        int nbVar= _element.size();
        int k;
        double g;
        DTLZObjectiveVector objVec(nbVar);

        k = nbVar - nbFun + 1;
        g = 0.0;
        for (unsigned i = nbVar - k + 1; i <= nbVar; i++)
            g += pow(_element[i-1]-0.5,2) - cos(20 * M_PI * (_element[i-1]-0.5));

        g = 100 *(k + g);

        for (unsigned i = 1; i <= nbFun; i++) {
            double f = 0.5 * (1 + g);
            for (unsigned j = nbFun - i; j >= 1; j--)
                f *= _element[j-1];

            if (i > 1)
                f *= 1 - _element[(nbFun - i + 1) - 1];

            objVec[i-1] = f;
        }

        _element.objectiveVector(objVec);
    }
}

