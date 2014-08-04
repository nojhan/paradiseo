/*
* <t-moeoEpsilonHyperboxArchive.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
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
// t-moeoEpsilonHyperboxArchive.cpp
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
	//!!!!!!!!!!!!VRAI TEST A FAIRE!!!!!!!!!!!!!!

    std::cout << "[moeoEpsilonHyperboxArchive]\t=>\t";

    std::cout << std::endl;

    // objective vectors
    ObjectiveVector obj;

    // population
    eoPop < Solution > pop;
    pop.resize(100);

    unsigned int o1=50;
    unsigned int o2=50;
    unsigned int o3=50;
    unsigned int o4=50;



    for(int i=0; i< pop.size()/2; i++){
//    	tmp=rng.uniform()*100;
    	obj[0]=o1;
    	obj[1]=o2;
//    	obj[0]=tmp;
//    	obj[1]=100-tmp;
    	pop[2*i].objectiveVector(obj);
    	obj[0]=o3;
    	obj[1]=o4;
//    	tmp=rng.uniform()*100;
//    	obj[0]=tmp;
//    	obj[1]=100-tmp;
    	pop[2*i + 1].objectiveVector(obj);
    	o1++;
    	o2--;
    	o3--;
    	o4++;
    }
//    pop.resize(4);
//    obj[0]=0;
//    obj[1]=100;
//    pop[0].objectiveVector(obj);
//    obj[0]=100;
//    obj[1]=0;
//    pop[1].objectiveVector(obj);
//    obj[0]=50;
//    obj[1]=50;
//    pop[2].objectiveVector(obj);
//    obj[0]=49;
//    obj[1]=50.5;
//    pop[3].objectiveVector(obj);

    std::vector < double > epsilon;
    epsilon.push_back(0.05);
    epsilon.push_back(0.05);

    // archive
    moeoEpsilonHyperboxArchive< Solution > arch(epsilon);

    ObjectiveVector nadir = arch.getNadir();
    ObjectiveVector ideal = arch.getIdeal();
    std::cout << "nadir: " << nadir << std::endl;
    std::cout << "ideal: " << ideal << std::endl;

    for(int i=0; i<pop.size() ; i++)
    	std::cout << pop[i].objectiveVector() << std::endl;

    for(int i=0; i<pop.size() ; i++){
    	arch(pop[i]);
//    nadir = arch.getNadir();
//    ideal = arch.getIdeal();
//    std::cout << "nadir: " << nadir << std::endl;
//    std::cout << "ideal: " << ideal << std::endl;
//    std::cout << "archive size: " << arch.size() << std::endl;
    }

    arch.filtre();

    std::cout << "archive size: " << arch.size() << std::endl;
    for (unsigned int i=0; i< arch.size(); i++)
		std::cout << arch[i].objectiveVector() << std::endl;

    std::cout << "nadir: " << nadir << std::endl;
    std::cout << "ideal: " << ideal << std::endl;
    //arch(pop);

    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
