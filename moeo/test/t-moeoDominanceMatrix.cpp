/*
* <t-moeoDominanceMatrix.cpp>
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
// t-moeoDominanceMatrix.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include <moeo>
#include <assert.h>
#include <set>
#include <iostream>
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
    std::cout << "[moeoDominanceMatrix]\n\n";

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

    moeoDominanceMatrix< Solution > matrix(true);

    moeoDominanceMatrix< Solution > matrix2(paretoComparator, false);

    //test operator() with 2 parameters
    matrix(pop,archive);

    //test result of matrix
    for (unsigned int i=0; i<7; i++)
        for (unsigned int j=0; j<3; j++)
            assert(!matrix[i][j]);
    assert(matrix[0][3]);
    assert(matrix[0][5]);
    assert(matrix[1][3]);
    assert(matrix[1][5]);
    assert(matrix[1][6]);
    assert(matrix[2][3]);
    assert(matrix[2][4]);
    assert(matrix[2][5]);
    assert(matrix[2][6]);
    assert(matrix[3][5]);
    assert(matrix[4][3]);
    assert(matrix[4][5]);
    assert(matrix[6][3]);
    assert(matrix[6][5]);
    assert(!matrix[0][4]);
    assert(!matrix[0][6]);
    assert(!matrix[1][4]);
    assert(!matrix[3][3]);
    assert(!matrix[3][4]);
    assert(!matrix[3][6]);
    assert(!matrix[4][4]);
    assert(!matrix[4][6]);
    assert(!matrix[5][3]);
    assert(!matrix[5][4]);
    assert(!matrix[5][5]);
    assert(!matrix[5][6]);
    assert(!matrix[6][4]);
    assert(!matrix[6][6]);

    //test methode count
    assert(matrix.count(0)==2);
    assert(matrix.count(1)==3);
    assert(matrix.count(2)==4);
    assert(matrix.count(3)==1);
    assert(matrix.count(4)==2);
    assert(matrix.count(5)==0);
    assert(matrix.count(6)==2);

    //test methode rank
    assert(matrix.rank(0)==0);
    assert(matrix.rank(1)==0);
    assert(matrix.rank(2)==0);
    assert(matrix.rank(3)==5);
    assert(matrix.rank(4)==1);
    assert(matrix.rank(5)==6);
    assert(matrix.rank(6)==2);

    //test operator() with one parameter
    matrix2(archive);

    assert(!matrix2[0][0]);
    assert(!matrix2[0][1]);
    assert(!matrix2[0][2]);
    assert(!matrix2[1][0]);
    assert(!matrix2[1][1]);
    assert(matrix2[1][2]);
    assert(!matrix2[2][0]);
    assert(!matrix2[2][1]);
    assert(!matrix2[2][2]);

    assert(matrix2.count(0)==0);
    assert(matrix2.count(1)==1);
    assert(matrix2.count(2)==0);

    assert(matrix2.rank(0)==0);
    assert(matrix2.rank(1)==0);
    assert(matrix2.rank(2)==1);

    std::set<int> hop;
    hop.insert(2);
    hop.insert(2);
    hop.insert(10);
    hop.insert(3);
    hop.insert(45);
    hop.insert(45);
    hop.insert(45);

    std::set<int>::iterator it=hop.begin();
    while (it!=hop.end()) {
        std::cout << *it << "\n";
        it++;
    }



    std::cout << "OK";
    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
