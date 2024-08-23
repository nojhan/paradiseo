/*
* <t-moeoExpBinaryIndicatorBasedFitnessAssignment.cpp>
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
// t-moeoExpBinaryIndicatorBasedFitnessAssignment.cpp
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
    std::cout << "[moeoExpBinaryIndicatorBasedFitnessAssignment]\t=>\t";

    // objective vectors
    ObjectiveVector obj0, obj1, obj2, obj3, obj4, obj5;
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

    // population
    eoPop < Solution > pop;
    pop.resize(6);
    pop[0].objectiveVector(obj0);
    pop[1].objectiveVector(obj1);
    pop[2].objectiveVector(obj2);
    pop[3].objectiveVector(obj3);
    pop[4].objectiveVector(obj4);
    pop[5].objectiveVector(obj5);

    // indicator
    moeoAdditiveEpsilonBinaryMetric < ObjectiveVector > indicator;

    // fitness assignment
    moeoExpBinaryIndicatorBasedFitnessAssignment< Solution > fitnessAssignment(indicator, 0.5);
    fitnessAssignment(pop);

    // pop[0]
    if ( (pop[0].fitness() > -1.56) || (pop[0].fitness() < -1.57) )
    {
        std::cout << "ERROR (bad fitness for pop[0])" << std::endl;
        return EXIT_FAILURE;
    }
    // pop[1]
    if ( (pop[1].fitness() > -2.40) || (pop[1].fitness() < -2.41) )
    {
        std::cout << "ERROR (bad fitness for pop[1])" << std::endl;
        return EXIT_FAILURE;
    }
    // pop[2]
    if ( (pop[2].fitness() > -1.51) || (pop[2].fitness() < -1.52) )
    {
        std::cout << "ERROR (bad fitness for pop[2])" << std::endl;
        return EXIT_FAILURE;
    }
    // pop[3]
    if ( (pop[3].fitness() > -9.38) || (pop[3].fitness() < -9.39) )
    {
        std::cout << "ERROR (bad fitness for pop[3]) " << std::endl;
        return EXIT_FAILURE;
    }
    // pop[4]
    if ( (pop[4].fitness() > -2.00) || (pop[4].fitness() < -2.01) )
    {
        std::cout << "ERROR (bad fitness for pop[4])" << std::endl;
        return EXIT_FAILURE;
    }
    // pop[5]
    if ( (pop[5].fitness() > -2.40) || (pop[5].fitness() < -2.41) )
    {
        std::cout << "ERROR (bad fitness for pop[5])" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
