/*
* <t-moeoWeakObjectiveVectorComparator.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
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
// t-moeoWeakObjectiveVectorComparator.cpp
//-----------------------------------------------------------------------------

#include <paradiseo/eo.h>
#include <paradiseo/moeo.h>

//-----------------------------------------------------------------------------

class ObjectiveVectorTraits : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int i)
    {
        return true;
    }
    static bool maximizing (int i)
    {
        return false;
    }
    static unsigned int nObjectives ()
    {
        return 2;
    }
};

typedef moeoRealObjectiveVector < ObjectiveVectorTraits > ObjectiveVector;

typedef MOEO < ObjectiveVector > Solution;

//-----------------------------------------------------------------------------

int main()
{
    std::cout << "[moeoWeakObjectiveVectorComparator]\t=>\t";

    // objective vectors
    ObjectiveVector obj1;
    obj1[0] = 2.0;
    obj1[1] = 5.0;
    ObjectiveVector obj2;
    obj2[0] = 2.0;
    obj2[1] = 5.0;
    ObjectiveVector obj3;
    obj3[0] = 3.0;
    obj3[1] = 5.0;
    ObjectiveVector obj4;
    obj4[0] = 3.0;
    obj4[1] = 6.0;

    // comparator
    moeoWeakObjectiveVectorComparator< ObjectiveVector > comparator;

    // obj1 dominated by obj2? YES
    if (! comparator(obj1, obj2))
    {
        std::cout << "ERROR (obj1 not dominated by obj2)" << std::endl;
        return EXIT_FAILURE;
    }

    // obj2 dominated by obj1? YES
    if (! comparator(obj2, obj1))
    {
        std::cout << "ERROR (obj2 not dominated by obj1)" << std::endl;
        return EXIT_FAILURE;
    }

    // obj1 dominated by obj3? NO
    if (comparator(obj1, obj3))
    {
        std::cout << "ERROR (obj1 dominated by obj3)" << std::endl;
        return EXIT_FAILURE;
    }

    // obj3 dominated by obj1? YES
    if (! comparator(obj3, obj1))
    {
        std::cout << "ERROR (obj3 not dominated by obj1)" << std::endl;
        return EXIT_FAILURE;
    }

    // obj3 dominated by obj4? NO
    if (comparator(obj3, obj4))
    {
        std::cout << "ERROR (obj3 dominated by obj4)" << std::endl;
        return EXIT_FAILURE;
    }

    // obj4 dominated by obj3? YES
    if (! comparator(obj4, obj3))
    {
        std::cout << "ERROR (obj4 not dominated by obj3)" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
