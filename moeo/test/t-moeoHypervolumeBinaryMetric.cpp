/*
* <t-moeoHypervolumeBinaryMetric.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2014
* 
* Suggested by Yann Semet
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
// t-moeoHypervolumeBinaryMetric.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include <moeo>

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
    std::cout << "[moeoHypervolumeBinaryMetric]\t=>\t";

    // objective vectors
    ObjectiveVector obj1;
    obj1[0] = 133.0;
    obj1[1] = 240.0;
    ObjectiveVector obj2;
    obj2[0] = 122.0;
    obj2[1] = 290.0;
    ObjectiveVector obj3;
    obj3[0] = 145.0;
    obj3[1] = 240.0;

    // indicator
    moeoHypervolumeBinaryMetric<ObjectiveVector> indicator(1.1);
    indicator.setup(122.0, 145.0, 0);
    indicator.setup(240.0, 290.0, 1);
    
    
    if ( (indicator(obj1, obj2) < 0.039525) || (indicator(obj1, obj2) > 0.039526))
    {
        std::cout << "ERROR (indicator-value I(obj1,obj2))" << std::endl;
        return EXIT_FAILURE;
    }
    
    if ( (indicator(obj2, obj1) < 0.51383) || (indicator(obj2, obj1) > 0.51384))
    {
        std::cout << "ERROR (indicator-value I(obj1,obj2))" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
