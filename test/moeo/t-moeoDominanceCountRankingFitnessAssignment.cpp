/*
* <t-moeoDominanceCountRankingFitnessAssignment.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Lille-Nord Europe, 2006-2008
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
// t-moeoDominanceCountRankingFitnessAssignment.cpp
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

typedef MOEO < ObjectiveVector, double, double > Solution;

//-----------------------------------------------------------------------------

int test(const eoPop < Solution >& _pop, const moeoUnboundedArchive < Solution >& _archive, double _pop0, double _pop1, double _pop2, double _pop3, double _arch0, double _arch1, double _arch2) {
    // pop[0]
    if (_pop[0].fitness() != _pop0)
    {
        std::cout << "ERROR (bad fitness for pop[0])" << std::endl;
        return EXIT_FAILURE;
    }
    // pop[1]
    if (_pop[1].fitness() != _pop1)
    {
        std::cout << "ERROR (bad fitness for pop[1])" << std::endl;
        return EXIT_FAILURE;
    }
    // pop[2]
    if (_pop[2].fitness() != _pop2)
    {
        std::cout << "ERROR (bad fitness for pop[2])" << std::endl;
        return EXIT_FAILURE;
    }
    // pop[3]
    if (_pop[3].fitness() != _pop3)
    {
        std::cout << "ERROR (bad fitness for pop[3]) " << std::endl;
        return EXIT_FAILURE;
    }
    // archive[0]
    if ((_arch0 <= 0 ) && (_archive[0].fitness() != _arch0))
    {
        std::cout << "ERROR (bad fitness archive[0])" << std::endl;
        return EXIT_FAILURE;
    }
    // archive[1]
    if ((_arch1 <= 0 ) && (_archive[1].fitness() != _arch1))
    {
        std::cout << "ERROR (bad fitness for archive[1])" << std::endl;
        return EXIT_FAILURE;
    }
    // archive[2]
    if ((_arch2 <= 0) && (_archive[2].fitness() != _arch2))
    {
        std::cout << "ERROR (bad fitness for archive[2])" << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}

int main()
{
    int res=EXIT_SUCCESS;
    std::cout << "[moeoDominanceCountRankingFitnessAssignment]\n";

    // objective vectors
    ObjectiveVector obj0, obj1, obj2, obj3, obj4, obj5, obj6;
    obj0[0] = 2;
    obj0[1] = 5;
    obj1[0] = 3;
    obj1[1] = 3;
    obj2[0] = 4;
    obj2[1] = 1;
    obj3[0] = 5;
    obj3[1] = 5;
    obj4[0] = 5;
    obj4[1] = 1;
    obj5[0] = 3;
    obj5[1] = 3;
    obj6[0] = 4;
    obj6[1] = 4;

    // population
    eoPop < Solution > pop;
    pop.resize(4);
    pop[0].objectiveVector(obj0);    // class 1
    pop[1].objectiveVector(obj1);    // class 1
    pop[2].objectiveVector(obj2);    // class 1
    pop[3].objectiveVector(obj3);    // class 3

    moeoUnboundedArchive < Solution > archive;
    archive.resize(3);
    archive[0].objectiveVector(obj4);
    archive[1].objectiveVector(obj5);
    archive[2].objectiveVector(obj6);

    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;

    // fitness assignment

    moeoDominanceCountRankingFitnessAssignment <Solution> fitnessAssignment;

    moeoDominanceCountRankingFitnessAssignment <Solution> fitnessAssignment2(archive);

    moeoDominanceCountRankingFitnessAssignment <Solution> fitnessAssignment3(paretoComparator,false);

    moeoDominanceCountRankingFitnessAssignment <Solution> fitnessAssignment4(paretoComparator, archive,false);

    std::cout << "Constructor without parameter => ";
    fitnessAssignment(pop);
    if (test(pop, archive, 0, 0, 0, -3, 1, 1, 1) == 0)
        std::cout << "OK" << std::endl;
    else
        res=EXIT_FAILURE;

    std::cout << "Constructor with archive passed in parameter => ";
    fitnessAssignment2(pop);
    for (unsigned int k=0; k<pop.size();k++)
        std::cout << "pop " << k << " : " << pop[k].fitness() << "\n";
    for (unsigned int k=0; k<archive.size();k++)
        std::cout << "archive " << k << " : " << archive[k].fitness() << "\n";

    if (test(pop, archive, 0, -14, 0, -13, -4, 0, -7) == 0)
        std::cout << "OK" << std::endl;
    else
        res=EXIT_FAILURE;

    return res;
}

//-----------------------------------------------------------------------------
