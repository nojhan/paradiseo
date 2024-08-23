/*
* <t-moeoHyperVolumeDifferenceMetric.cpp>
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
// t-moeoHyperVolumeDifferenceMetric.cpp
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
        return 3;
    }
};

typedef moeoRealObjectiveVector < ObjectiveVectorTraits2 > ObjectiveVector2;

//-----------------------------------------------------------------------------

int main()
{
    std::cout << "[moeoHyperVolumeDifferenceMetric] => \n";

    // objective vectors
    std::vector < ObjectiveVector > set1;
    std::vector < ObjectiveVector > set2;

    //test normalisation
    set1.resize(3);
    set2.resize(3);

    //test case
    set1[0][0] = 5;
    set1[0][1] = 1;

    set1[1][0] = 2;
    set1[1][1] = 4;

    set1[2][0] = 1;
    set1[2][1] = 5;

    set2[0][0] = 4;
    set2[0][1] = 2;

    set2[1][0] = 3;
    set2[1][1] = 3;

    set2[2][0] = 1;
    set2[2][1] = 6;

    moeoHyperVolumeDifferenceMetric < ObjectiveVector> metric(true, 2);

	std::vector < eoRealInterval > bounds;

	metric.setup(set1, set2);
	bounds = metric.getBounds();

    std::cout << "\t>test normalization =>";
    assert(bounds[0].minimum()==1.0);
    assert(bounds[0].maximum()==5.0);
    assert(bounds[0].range()==4.0);

    assert(bounds[1].minimum()==1.0);
    assert(bounds[1].maximum()==6.0);
    assert(bounds[1].range()==5.0);

	std::cout << " OK\n";

	//test calculation of difference hypervolume
	moeoHyperVolumeDifferenceMetric < ObjectiveVector> metric2(false, 2);
	std::cout << "\t>test difference without normalization and a coefficient rho=>";
	double difference=metric2(set1,set2);
	assert(difference==5.0);
	std::cout << " OK\n";

	moeoHyperVolumeDifferenceMetric < ObjectiveVector> metric3(true, 1.1);
	double tolerance = 1e-10;
	std::cout << "\t>test difference with normalization and coefficient rho=>";
	difference=metric3(set1,set2);
	assert( (difference < (0.02 + tolerance)) && (difference > (0.02 - tolerance)));
	std::cout << " OK\n";

	ObjectiveVector ref_point;
	ref_point[0]= 10;
	ref_point[1]= 12;
	moeoHyperVolumeDifferenceMetric < ObjectiveVector> metric4(false, ref_point);
	std::cout << "\t>test difference without normalization and a ref_point=>";
	difference=metric4(set1,set2);
	assert(difference==5.0);
	std::cout << " OK\n";

	ref_point[0]= 1.1;
	ref_point[1]= 1.1;
	moeoHyperVolumeDifferenceMetric < ObjectiveVector> metric5(true, ref_point);
	std::cout << "\t>test difference with normalization and a ref_point=>";
	difference=metric5(set1,set2);
	assert( (difference < (0.02 + tolerance)) && (difference > (0.02 - tolerance)));
	std::cout << " OK\n";




    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
