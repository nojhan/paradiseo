/*
* <t-moeoSubNeighborhoodExplorer.cpp>
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
// t-moeoSubNeighborhoodExplorer.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include <mo>
#include <moeo>
#include <assert.h>
#include <set>
#include <iostream>
#include <moeoSimpleSubNeighborhoodExplorer.h>
#include <moeoTestClass.h>
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------

int main()
{
    std::cout << "[moeoSubNeighborhoodExplorer]\n\n";

    // objective vectors
    ObjectiveVector obj0, obj1;
    obj0[0]=0;
    obj0[1]=1;
    obj1[0]=10;
    obj1[1]=11;
    eoPop < Solution > src;
    eoPop < Solution > dest;
    std::vector < unsigned int > select;
    src.resize(5);
    src[2].objectiveVector(obj0);
    src[4].objectiveVector(obj1);
    src[0].flag(0);
    src[1].flag(0);
    src[2].flag(0);
    src[3].flag(0);
    src[4].flag(0);
    dest.resize(0);
    select.resize(2);
    select[0]=2;
    select[1]=4;

    testMoveInit init;
    testMoveNext next;
    testMoveIncrEval incr;

    moeoSimpleSubNeighborhoodExplorer < testMove > explorer(init, next, incr, 6);

    explorer(src, select, dest);
    assert(dest.size()==10);
    assert(src[0].flag()==0);
    assert(src[1].flag()==0);
    assert(src[2].flag()==1);
    assert(src[3].flag()==0);
    assert(src[4].flag()==1);
    for(int i=0 ; i< 10 ; i++)
    	assert(dest[i].flag()==0);
    assert(dest[0].objectiveVector()[0]==2);
    assert(dest[1].objectiveVector()[1]==3);
    assert(dest[2].objectiveVector()[0]==2);
    assert(dest[3].objectiveVector()[1]==3);
    assert(dest[4].objectiveVector()[0]==2);

    assert(dest[5].objectiveVector()[0]==12);
    assert(dest[6].objectiveVector()[1]==13);
    assert(dest[7].objectiveVector()[0]==12);
    assert(dest[8].objectiveVector()[1]==13);
    assert(dest[9].objectiveVector()[0]==12);

    src[2].flag(0);
    src[4].flag(0);
    dest.resize(0);

    moeoSimpleSubNeighborhoodExplorer < testMove > explorer2(init, next, incr, 3);

    explorer2(src, select, dest);
    assert(dest.size()==6);
    assert(src[0].flag()==0);
    assert(src[1].flag()==0);
    assert(src[2].flag()==0);
    assert(src[3].flag()==0);
    assert(src[4].flag()==0);
    for(int i=0 ; i< 6 ; i++)
    	assert(dest[i].flag()==0);
    assert(dest[0].objectiveVector()[0]==2);
    assert(dest[1].objectiveVector()[1]==3);
    assert(dest[2].objectiveVector()[0]==2);

    assert(dest[3].objectiveVector()[0]==12);
    assert(dest[4].objectiveVector()[1]==13);
    assert(dest[5].objectiveVector()[0]==12);


    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;

}

//-----------------------------------------------------------------------------
