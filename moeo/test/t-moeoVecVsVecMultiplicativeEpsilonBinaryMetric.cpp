/*
* <t-moeoVecVsVecMultiplicativeEpsilonBinaryMetric.cpp>
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
// t-moeoVecVsVecMultiplicativeEpsilonBinaryMetric.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include <moeo>
#include <assert.h>

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

class ObjectiveVectorTraits2 : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int i)
    {
        if (i==0)
            return true;
        else
            return false;
    }
    static bool maximizing (int i)
    {
        if (i==0)
            return false;
        else
            return true;

    }
    static unsigned int nObjectives ()
    {
        return 2;
    }
};

typedef moeoRealObjectiveVector < ObjectiveVectorTraits2 > ObjectiveVector2;

//-----------------------------------------------------------------------------

int main()
{
    std::cout << "[moeoVecVsVecMultiplicativeEpsilonBinaryMetric] => \n";

    double eps;

    // objective vectors
    std::vector < ObjectiveVector > set1;
    std::vector < ObjectiveVector > set2;
    std::vector < ObjectiveVector > set3;
    std::vector < ObjectiveVector > set4;

    set1.resize(5);
    set2.resize(4);
    set3.resize(5);
    set4.resize(3);

    set1[0][0] = 4;
    set1[0][1] = 7;

    set1[1][0] = 5;
    set1[1][1] = 6;

    set1[2][0] = 7;
    set1[2][1] = 5;

    set1[3][0] = 8;
    set1[3][1] = 4;

    set1[4][0] = 9;
    set1[4][1] = 2;

    set2[0][0] = 4;
    set2[0][1] = 7;

    set2[1][0] = 5;
    set2[1][1] = 6;

    set2[2][0] = 7;
    set2[2][1] = 5;

    set2[3][0] = 8;
    set2[3][1] = 4;

    set3[0][0] = 10;
    set3[0][1] = 4;

    set3[1][0] = 9;
    set3[1][1] = 5;

    set3[2][0] = 8;
    set3[2][1] = 6;

    set3[3][0] = 7;
    set3[3][1] = 7;

    set3[4][0] = 6;
    set3[4][1] = 8;

    set4[0][0] = 3;
    set4[0][1] = 1;

    set4[1][0] = 2;
    set4[1][1] = 2;

    set4[2][0] = 1;
    set4[2][1] = 3;



    moeoVecVsVecMultiplicativeEpsilonBinaryMetric < ObjectiveVector > metric;

    std::cout << "\t>Ieps(set1, set2) => ";
    eps = metric(set1, set2);
    assert(eps == 1.0);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set2, set1) => ";
    eps = metric(set2, set1);
    assert(eps == 2.0);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set1, set3) => ";
    eps = metric(set1, set3);
    assert(eps == 0.9);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set3, set1) => ";
    eps = metric(set3, set1);
    assert(eps == 2.0);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set1, set4) => ";
    eps = metric(set1, set4);
    assert(eps == 4.0);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set4, set1) => ";
    eps = metric(set4, set1);
    assert(eps == 0.5);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set2, set3) => ";
    eps = metric(set2, set3);
    assert(eps == 1.0);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set3, set2) => ";
    eps = metric(set3, set2);
    assert(eps == 1.5);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set2, set4) => ";
    eps = metric(set2, set4);
    assert(eps == 4.0);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set4, set2) => ";
    eps = metric(set4, set2);
    assert(eps == 3.0/7.0);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set3, set4) => ";
    eps = metric(set3, set4);
    assert(eps == 6.0);
    std::cout << "OK\n";

    std::cout << "\t>Ieps(set4, set3) => ";
    eps = metric(set4, set3);
    assert(eps == 1.0/3.0);
    std::cout << "OK\n";

    set1[0][0] = 0;
    set3[0][1] = -1;

    std::cout << "\tError test: elements with a differents sign => ";
    try{
        eps = metric(set4, set3);
        return EXIT_FAILURE;
    }
    catch (char const* e){
        std::cout << "OK\n";
    }

    std::cout << "\tError test: an element = 0 => ";
    try{
        eps = metric(set1, set2);
        return EXIT_FAILURE;
    }
    catch (char const* e){
        std::cout << "Ok\n";
    }

    //test with maximization
    moeoVecVsVecMultiplicativeEpsilonBinaryMetric < ObjectiveVector2 > metric2;

    std::vector < ObjectiveVector2 > set5;
    std::vector < ObjectiveVector2 > set6;


    set5.resize(3);
    set6.resize(4);

    set5[0][0] = 1;
    set5[0][1] = 3;

    set5[1][0] = 2;
    set5[1][1] = 2;

    set5[2][0] = 3;
    set5[2][1] = 1;

    set6[0][0] = 5;
    set6[0][1] = 2;

    set6[1][0] = 4;
    set6[1][1] = 3;

    set6[2][0] = 3;
    set6[2][1] = 4;

    set6[3][0] = 2;
    set6[3][1] = 5;

    std::cout << "\t1 Maximazing objectif test => ";
    eps = metric2(set5, set6);
    assert(eps == 5.0/3.0);
    std::cout << "Ok\n";

    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
