/*
<t-moRestartPerturb.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <iostream>
#include <cstdlib>
#include <cassert>

#include <paradiseo/mo/perturb/moRestartPerturb.h>
#include <paradiseo/mo/problems/permutation/moShiftNeighbor.h>
#include <paradiseo/problems/eval/queenEval.h>

#include <paradiseo/eo/eoInt.h>
#include <paradiseo/eo/eoInit.h>

typedef eoInt<unsigned int> QUEEN;

class dummyInit : public eoInit<QUEEN>
{
public:
    dummyInit(unsigned int _size):size(_size) {}

    void operator()(QUEEN& _sol) {
        _sol.resize(0);
        for (unsigned int i=0; i<size; i++)
            _sol.push_back(i);
        _sol.invalidate();
    }


private:
    unsigned int size;

};

int main() {

    std::cout << "[t-moRestartPerturb] => START" << std::endl;

    QUEEN queen;
    moShiftNeighbor<QUEEN> n;

    dummyInit initializer(4);

    queenEval<QUEEN> eval;

    moRestartPerturb<moShiftNeighbor<QUEEN> > test(initializer, eval, 3);

    queen.resize(4);
    queen[0]=1;
    queen[1]=2;
    queen[2]=0;
    queen[3]=3;

    test.init(queen);

    test(queen);
    assert(queen[0]==1 && queen[1]==2 && queen[2]==0 && queen[3]==3);
    test.update(queen,n); //first noMove
    test(queen);
    assert(queen[0]==1 && queen[1]==2 && queen[2]==0 && queen[3]==3);
    test.update(queen,n); //second noMove
    test(queen);
    assert(queen[0]==1 && queen[1]==2 && queen[2]==0 && queen[3]==3);
    test.update(queen,n); //third noMove
    test(queen);//here the perturb should be called
    assert(queen[0]==0 && queen[1]==1 && queen[2]==2 && queen[3]==3);

    queen[0]=1;
    queen[1]=2;
    queen[2]=0;
    queen[3]=3;

    //Retry the same test to verify counter is been reinit to 0

    test(queen);
    assert(queen[0]==1 && queen[1]==2 && queen[2]==0 && queen[3]==3);
    test.update(queen,n); //first noMove
    test(queen);
    assert(queen[0]==1 && queen[1]==2 && queen[2]==0 && queen[3]==3);
    test.update(queen,n); //second noMove
    test(queen);
    assert(queen[0]==1 && queen[1]==2 && queen[2]==0 && queen[3]==3);
    test.update(queen,n); //third noMove
    test(queen); //here the perturb should be called
    assert(queen[0]==0 && queen[1]==1 && queen[2]==2 && queen[3]==3);

    std::cout << "[t-moRestartPerturb] => OK" << std::endl;

    return EXIT_SUCCESS;
}

