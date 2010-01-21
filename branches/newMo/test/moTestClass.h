#ifndef _moTestClass_h
#define _moTestClass_h

#include <EO.h>
#include <eoEvalFunc.h>
#include <neighborhood/moNeighbor.h>
#include <neighborhood/moBackableNeighbor.h>

typedef EO<int> Solution;

class moDummyNeighbor : public moNeighbor<Solution,int>{
public:
    virtual void move(Solution & _solution){}
};

class moDummyBackableNeighbor : public moBackableNeighbor<Solution,int>{
public:
    virtual void move(Solution & _solution){}
    virtual void moveBack(Solution & _solution){}
};

class moDummyEval: public eoEvalFunc<Solution>{
public:
	void operator()(Solution& _sol){
		if(_sol.invalid())
			_sol.fitness(100);
		else
			_sol.fitness(_sol.fitness()+50);
	}
};

#endif
