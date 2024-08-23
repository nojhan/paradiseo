/*
* <t-moeoSPEA2Archive.cpp>
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
// t-moeoSPEA2Archive.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include <moeo>

//-----------------------------------------------------------------------------

class ObjectiveVectorTraits : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int /*i*/)
    {
        return true;
    }
    static bool maximizing (int /*i*/)
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

int main()
{
    std::cout << "[moeoSPEA2Archive] => ";

    float tmp1=0;
    float tmp2=0;

    // objective vectors
    ObjectiveVector obj0, obj1, obj2, obj3;
    obj0[0] = 2;
    obj0[1] = 1;
    obj1[0] = 1;
    obj1[1] = 2;
    obj2[0] = 0;
    obj2[1] = 4;
    obj3[0] = 5;
    obj3[1] = 0;

    // population
    eoPop < Solution > pop;
    pop.resize(4);
    pop[0].objectiveVector(obj0);
    pop[1].objectiveVector(obj1);
    pop[2].objectiveVector(obj2);
    pop[3].objectiveVector(obj3);

    //distance
    moeoEuclideanDistance<Solution> dist;

    //objective vector comparator
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;

    //comparator
    moeoFitnessThenDiversityComparator < Solution > indiComparator;

    //archive
    moeoSPEA2Archive<Solution> archive(dist,3);

    moeoSPEA2Archive<Solution> archive2;

    moeoSPEA2Archive<Solution> archive3(indiComparator,1);

    moeoSPEA2Archive<Solution> archive4(paretoComparator);

    moeoSPEA2Archive<Solution> archive5(indiComparator, dist, paretoComparator, 3);

    // diversity assignment
    moeoNearestNeighborDiversityAssignment<Solution> diversityassignement(dist,archive, 2);
    diversityassignement(pop);

    // fitness assignment
    moeoDominanceCountRankingFitnessAssignment <Solution> fitnessassignement(archive,false);
    fitnessassignement(pop);

//-----------------------------------------------------------------------------
//first test archive: the archive is empty -> the best element of the pop must be copy in the archive.


    archive(pop);

    //archive[0]
    tmp1=archive[0].diversity();
    tmp2=-1/(2+sqrt(20.0));
    if ( (tmp1 != tmp2)  || (archive[0].fitness() != 0.0) )
    {
        std::cout << "ERROR (Test 1: bad result for archive[0])" << std::endl;
        return EXIT_FAILURE;
    }
    //archive[1]
    tmp1=archive[1].diversity();
    tmp2=-1/(2+sqrt(13.0));
    if ( (tmp1 != tmp2) || (archive[1].fitness() != 0.0) )
    {
        std::cout << "ERROR (Test 1: bad result for archive[1])" << std::endl;
        return EXIT_FAILURE;
    }
    //archive[2]
    tmp1=archive[2].diversity();
    tmp2=-1/(2+sqrt(10.0));
    if ( (tmp1 != tmp2) || (archive[2].fitness() != 0.0) )
    {
        std::cout << "ERROR (Test 1: bad result for archive[2])" << std::endl;
        return EXIT_FAILURE;
    }
//-----------------------------------------------------------------------------

    obj0[0] = 3;
    obj0[1] = 1;
    obj1[0] = 2;
    obj1[1] = 2;
    obj3[0] = 1;
    obj3[1] = 3;
    pop[0].objectiveVector(obj0);
    pop[1].objectiveVector(obj1);
    pop[3].objectiveVector(obj3);

    diversityassignement(pop);
    fitnessassignement(pop);

    //-----------------------------------------------------------------------------
    //second test archive : (there are more elements with fitness=0 than the size of the archive)
    //depends of the good result of the first test
    archive(pop);
    //archive[0]
    tmp1=archive[0].diversity();
    tmp2=-1/3.0;
    if ( (tmp1 != tmp2) || (archive[0].fitness() != 0.0) )
    {
        std::cout << "ERROR (Test 2: bad result for archive[0])" << std::endl;
        return EXIT_FAILURE;
    }
    //archive[1]
    tmp1=archive[1].diversity();
    tmp2=-1/(2+sqrt(2.0));
    if ( (tmp1 != tmp2) || (archive[1].fitness() != 0.0) )
    {
        std::cout << "ERROR (Test 2: bad result for archive[1])" << std::endl;
        return EXIT_FAILURE;
    }
    //archive[2]
    tmp1=archive[2].diversity();
    tmp2=-1/(2+sqrt(10.0));
    if ( (tmp1 != tmp2) || (archive[2].fitness() != 0.0) )
    {
        std::cout << "ERROR (Test 2: bad result for archive[2])" << std::endl;
        return EXIT_FAILURE;
    }
    //-----------------------------------------------------------------------------

    obj0[0] = 5;
    obj0[1] = 5;
    obj1[0] = 4;
    obj1[1] = 4;
    obj2[0] = 4;
    obj2[1] = 5;
    obj3[0] = 4;
    obj3[1] = 0;

    pop[0].objectiveVector(obj0);
    pop[1].objectiveVector(obj1);
    pop[2].objectiveVector(obj2);
    pop[3].objectiveVector(obj3);

    diversityassignement(pop);
    fitnessassignement(pop);

    //-----------------------------------------------------------------------------
    //third test archive : a pop element with fitness=0 replace a worst archive element
    //depends of the good result of the two first tests

    archive(pop);

    //archive[0]
    tmp1=archive[0].diversity();
    tmp2=-1/(2+sqrt(16.0));
    if ( (tmp1 != tmp2) || (archive[0].fitness() != 0.0) )
    {
        std::cout << "ERROR (Test 3: bad result for archive[0])" << std::endl;
        return EXIT_FAILURE;
    }
    //archive[1]
    tmp1=archive[1].diversity();
    tmp2=-1/(2+sqrt(10.0));
    if ( (tmp1 != tmp2) || (archive[1].fitness() != -0.0) )
    {
        std::cout << "ERROR (Test 3: bad result for archive[1])" << std::endl;
        return EXIT_FAILURE;
    }
    //archive[2]
    tmp1=archive[2].diversity();
    tmp2=-1/(2+sqrt(5.0));
    if ( (tmp1 != tmp2) || (archive[2].fitness() != 0.0) )
    {
        std::cout << "ERROR (Test 3: bad result for archive[2])" << std::endl;
        return EXIT_FAILURE;
    }
    //-----------------------------------------------------------------------------
    archive(pop[0]);

    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
