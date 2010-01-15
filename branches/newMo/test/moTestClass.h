#ifndef _moTestClass_h
#define _moTestClass_h

#include <EO.h>
#include <neighborhood/moNeighbor.h>

typedef EO<int> Solution;

class moDummyNeighbor : public moNeighbor<Solution,int>{

    virtual void eval(Solution & solution){}

    virtual void move(Solution & solution){}
};

#endif
