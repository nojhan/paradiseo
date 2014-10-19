/*
* <t-moeoDetArchiveSelect.cpp>
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
// t-moeoDetArchiveSelect..cpp
//-----------------------------------------------------------------------------

#include <paradiseo/eo.h>
#include <paradiseo/moeo.h>
#include <cassert>


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

int main()
{
    std::cout << "[moeoDetArchiveSelect] => \n";

	moeoUnboundedArchive <Solution> archive;
	Solution sol1, sol2, sol3, sol4, sol5;
	ObjectiveVector obj1, obj2, obj3, obj4, obj5;
	obj1[0]=10;
	obj1[1]=0;
	obj2[0]=9;
	obj2[1]=1;
	obj3[0]=8;
	obj3[1]=2;
	obj4[0]=7;
	obj4[1]=3;
	obj5[0]=6;
	obj5[1]=4;

	sol1.objectiveVector(obj1);
	sol2.objectiveVector(obj2);
	sol3.objectiveVector(obj3);
	sol4.objectiveVector(obj4);
	sol5.objectiveVector(obj5);

	archive(sol1);
	archive(sol2);
	archive(sol3);
	archive(sol4);
	archive(sol5);
	assert(archive.size() == 5);

	//archive.printOn(std::cout);

	eoPop <Solution> source, dest;

	// test with max > archive size
	moeoDetArchiveSelect <Solution> select1(archive, 10);
	select1(source, dest);
	for(unsigned int i=0; i< archive.size(); i++){
		assert(dest[i].objectiveVector()[0]==archive[i].objectiveVector()[0]);
		assert(dest[i].objectiveVector()[1]==archive[i].objectiveVector()[1]);
	}

	//test with a max < archive size
	dest.resize(0);
	moeoDetArchiveSelect <Solution> select2(archive, 3);
	select2(source, dest);
	assert(dest.size()==3);

	//test with archive size < min
	dest.resize(0);
	moeoDetArchiveSelect <Solution> select3(archive, 100, 10);
	select3(source, dest);
	for(int i=0; i< 10; i++){
		assert(dest[i].objectiveVector()[0]==archive[i%archive.size()].objectiveVector()[0]);
		assert(dest[i].objectiveVector()[1]==archive[i%archive.size()].objectiveVector()[1]);
	}

	//test with bad value
	dest.resize(0);
	moeoDetArchiveSelect <Solution> select4(archive, 10, 11);
	select4(source, dest);
	assert(dest.size()==0);

	std::cout << " OK\n";
    return EXIT_SUCCESS;
}
