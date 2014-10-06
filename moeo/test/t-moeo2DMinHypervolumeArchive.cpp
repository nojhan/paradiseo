/*
* <t-moeo2DMinHypervolumeArchive.cpp>
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
*
*/
//-----------------------------------------------------------------------------
// t-moeo2DMinHypervolumeArchive.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include <moeo>
#include <cassert>

#include<archive/moeo2DMinHyperVolumeArchive.h>

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

typedef moeoBitVector < ObjectiveVector > Solution;

//-----------------------------------------------------------------------------

int main()
{
    std::cout << "[moeo2DMinHyperVolumeArchive]\t=>\t";

    // objective vectors
    ObjectiveVector obj;
    obj[0] = 7;
    obj[1] = 15;


    // population
    eoPop < Solution > pop;
    pop.resize(0);


    //Solutions
    Solution a;
    a.objectiveVector(obj);

    // archive
    moeo2DMinHypervolumeArchive<Solution> arch(5,1000);

    //test archive
    pop.push_back(a);

    obj[0]=10;
    obj[1]=14;
    a.objectiveVector(obj);
    pop.push_back(a);


    obj[0]=8;
    obj[1]=14.5;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=6;
    obj[1]=15;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=14;
    obj[1]=13;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=12;
    obj[1]=13.5;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=14.5;
    obj[1]=12.5;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=14;
    obj[1]=13;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=11.5;
    obj[1]=11.5;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=5.5;
    obj[1]=12.5;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=3.5;
    obj[1]=13.5;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=15;
    obj[1]=3.5;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=13.5;
    obj[1]=8;
    a.objectiveVector(obj);
    pop.push_back(a);

    obj[0]=1;
    obj[1]=15;
    a.objectiveVector(obj);
    pop.push_back(a);

    for(int i=0; i< pop.size(); i++){
    	arch(pop[i]);
		arch.print();
		std::cout << std::endl;
    }

    moeo2DMinHypervolumeArchive<Solution>::iterator it;
    it=arch.begin();
    assert(it->objectiveVector()[0]==15);
    assert(it->objectiveVector()[1]==3.5);
    assert(it->fitness()==1000);
    it++;
    assert(it->objectiveVector()[0]==13.5);
    assert(it->objectiveVector()[1]==8);
    assert(it->fitness()==6.75);
    it++;
    assert(it->objectiveVector()[0]==5.5);
    assert(it->objectiveVector()[1]==12.5);
    assert(it->fitness()==8);
    it++;
    assert(it->objectiveVector()[0]==3.5);
    assert(it->objectiveVector()[1]==13.5);
    assert(it->fitness()==3);
    it++;
    assert(it->objectiveVector()[0]==1);
    assert(it->objectiveVector()[1]==15);
    assert(it->fitness()==1000);


    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
}



//-----------------------------------------------------------------------------
