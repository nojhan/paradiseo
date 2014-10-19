/*
* <t-moeoNearestNeighborDiversityAssignment.cpp>
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
// t-moeoNearestNeighborDiversityAssignment.cpp
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

int main()
{
    std::cout << "[moeoNearestNeighborDiversityAssignment]\n";

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

    //archive
    moeoUnboundedArchive < Solution > archive;
    archive.resize(3);
    archive[0].objectiveVector(obj4);
    archive[1].objectiveVector(obj5);
    archive[2].objectiveVector(obj6);

    //distance
    moeoEuclideanDistance<Solution> dist;

    // diversity assignment
    moeoNearestNeighborDiversityAssignment<Solution> diversityAssignment;

    moeoNearestNeighborDiversityAssignment<Solution> diversityAssignment2(archive, 2);

    moeoNearestNeighborDiversityAssignment<Solution> diversityAssignment3(dist);

    moeoNearestNeighborDiversityAssignment<Solution> diversityAssignment4(dist,archive, 2);

    int res=EXIT_SUCCESS;

    for (unsigned int i=0; i<2; i++) {
        float tmp1=0;
        float tmp2=0;

        if (i==0) {
            diversityAssignment(pop);
            // pop[0]
            tmp1=pop[0].diversity();
            tmp2=-1/(2+sqrt(5.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity for pop[0])" << std::endl;
                res=EXIT_FAILURE;
            }
            // pop[1]
            tmp1=pop[1].diversity();
            tmp2=-1/(2+sqrt(5.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity for pop[1])" << std::endl;
                res=EXIT_FAILURE;
            }
            // pop[2]
            tmp1=pop[2].diversity();
            tmp2=-1/(2+sqrt(5.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity for pop[2])" << std::endl;
                res=EXIT_FAILURE;
            }
            // pop[3]
            tmp1=pop[3].diversity();
            tmp2=-1/(2+sqrt(8.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity for pop[3]) " << std::endl;
                res=EXIT_FAILURE;
            }
        }
        else {
            diversityAssignment4(pop);
            // pop[0]
            tmp1=pop[0].diversity();
            tmp2=-1/(2+sqrt(5.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity for pop[0])" << std::endl;
                res=EXIT_FAILURE;
            }
            // pop[1]
            tmp1=pop[1].diversity();
            tmp2=-1/(2+sqrt(2.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity for pop[1])" << std::endl;
                res=EXIT_FAILURE;
            }
            // pop[2]
            tmp1=pop[2].diversity();
            tmp2=-1/(2+sqrt(5.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity for pop[2])" << std::endl;
                res=EXIT_FAILURE;
            }
            // pop[3]
            tmp1=pop[3].diversity();
            tmp2=-1/(2+sqrt(8.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity for pop[3]) " << std::endl;
                res=EXIT_FAILURE;
            }
            // archive[0]
            tmp1=archive[0].diversity();
            tmp2=-1/(2+sqrt(8.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity archive[0])" << std::endl;
                res=EXIT_FAILURE;
            }
            // archive[1]
            tmp1=archive[1].diversity();
            tmp2=-1/(2+sqrt(2.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity for archive[1])" << std::endl;
                res=EXIT_FAILURE;
            }
            // archive[2]
            tmp1=archive[2].diversity();
            tmp2=-1/(2+sqrt(2.0));
            if (tmp1 != tmp2)
            {
                std::cout << "ERROR (bad diversity for archive[2])" << std::endl;
                res=EXIT_FAILURE;
            }

        }

    }

    std::cout << "OK" << std::endl;
    return res;
}

//-----------------------------------------------------------------------------
