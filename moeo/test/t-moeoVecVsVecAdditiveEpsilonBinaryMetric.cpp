/*
* <t-moeoVecVsVecAdditiveEpsilonBinaryMetric.cpp>
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
// t-moeoVecVsVecAdditiveEpsilonBinaryMetric.cpp
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
    std::cout << "[moeoVecVsVecAdditiveEpsilonBinaryMetric] => \n";

    double eps;

    // objective vectors
    std::vector < ObjectiveVector > set1;
    std::vector < ObjectiveVector > set2;

    //test 1: set2 dominates set1
    set1.resize(4);
    set2.resize(4);

    set1[0][0] = 6;
    set1[0][1] = 3;

    set1[1][0] = 5;
    set1[1][1] = 4;

    set1[2][0] = 4;
    set1[2][1] = 5;

    set1[3][0] = 2;
    set1[3][1] = 7;

    set2[0][0] = 1;
    set2[0][1] = 5;

    set2[1][0] = 2;
    set2[1][1] = 3;

    set2[2][0] = 3;
    set2[2][1] = 2;

    set2[3][0] = 4;
    set2[3][1] = 1;

    moeoVecVsVecAdditiveEpsilonBinaryMetric < ObjectiveVector > metric(false);

    eps = metric(set1, set2);
    assert(eps == 2.0);
    std::cout << "\t>test1 => OK\n";
    //end test1

    //test2: set1 dominates set2
    set2.resize(3);

    set1[0][0] = 0;
    set1[0][1] = 6;

    set1[1][0] = 1;
    set1[1][1] = 3;

    set1[2][0] = 3;
    set1[2][1] = 1;

    set1[3][0] = 6;
    set1[3][1] = 0;

    set2[0][0] = 1;
    set2[0][1] = 5;

    set2[1][0] = 3;
    set2[1][1] = 3;

    set2[2][0] = 5;
    set2[2][1] = 2;

    eps = metric(set1, set2);
    assert(eps == 0.0);
    std::cout << "\t>test2 => OK\n";
    //end test2

    set2.resize(4);
    //test3: no dominance
    set1[0][0] = 7;
    set1[0][1] = 1;

    set1[1][0] = 6;
    set1[1][1] = 4;

    set1[2][0] = 3;
    set1[2][1] = 4;

    set1[3][0] = 2;
    set1[3][1] = 7;

    set2[0][0] = 8;
    set2[0][1] = 2;

    set2[1][0] = 5;
    set2[1][1] = 3;

    set2[2][0] = 4;
    set2[2][1] = 5;

    set2[3][0] = 1;
    set2[3][1] = 6;

    eps = metric(set1, set2);
    assert(eps == 1.0);
    std::cout << "\t>test3 => OK\n";
    //end test3

    //test bounds
    std::vector < eoRealInterval > bounds;

    moeoVecVsVecAdditiveEpsilonBinaryMetric < ObjectiveVector > metric2;
    metric2.setup(set1, set2);
    bounds = metric2.getBounds();

    assert(bounds[0].minimum()==1.0);
    assert(bounds[0].maximum()==8.0);
    assert(bounds[0].range()==7.0);

    assert(bounds[1].minimum()==1.0);
    assert(bounds[1].maximum()==7.0);
    assert(bounds[1].range()==6.0);
    std::cout << "\t>test normalization => OK\n";
    //end test bounds

    std::vector < ObjectiveVector2 > set3;
    std::vector < ObjectiveVector2 > set4;
    moeoVecVsVecAdditiveEpsilonBinaryMetric < ObjectiveVector2 > metric3(false);

    //test 1: set2 dominates set1
    set3.resize(2);
    set4.resize(2);

    set3[0][0] = 6;
    set3[0][1] = 3;

    set3[1][0] = 5;
    set3[1][1] = 4;

    set4[0][0] = 1;
    set4[0][1] = 5;

    set4[1][0] = 2;
    set4[1][1] = 3;

    std::cout << "\t>test with maximization =>";
    eps = metric3(set3, set4);
    assert(eps==4.0);
    std::cout << "OK\n";

    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
