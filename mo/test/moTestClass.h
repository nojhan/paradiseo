/*
 <moTestClass.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

 As a counterpart to the access to the source code and  rights to copy,
 modify and redistribute granted by the license, users are provided only
 with a limited warranty  and the software's author,  the holder of the
 economic rights,  and the successive licensors  have only  limited liability.

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

#ifndef _moTestClass_h
#define _moTestClass_h

#include <EO.h>
#include <eoEvalFunc.h>
#include <neighborhood/moNeighbor.h>
#include <neighborhood/moBackableNeighbor.h>
#include <neighborhood/moNeighborhood.h>
#include <neighborhood/moRndNeighborhood.h>
#include <eval/moEval.h>

#include <ga/eoBit.h>
#include <eoScalarFitness.h>
#include <neighborhood/moOrderNeighborhood.h>
#include <problems/bitString/moBitNeighbor.h>
#include <neighborhood/moIndexNeighbor.h>

#include <utils/eoMonitor.h>
#include <utils/eoUpdater.h>

#include <eoInit.h>

typedef eoBit<eoMinimizingFitness> bitVector;
typedef moBitNeighbor<eoMinimizingFitness> bitNeighbor;

class moDummyRndNeighborhood: public moOrderNeighborhood<bitNeighbor> ,
		public moRndNeighborhood<bitNeighbor> {
public:
	moDummyRndNeighborhood(unsigned int a) :
		moOrderNeighborhood<bitNeighbor> (a) {
	}
};

typedef moDummyRndNeighborhood bitNeighborhood;

typedef EO<int> Solution;

class moDummyNeighborTest: public moNeighbor<Solution> {
public:
	virtual void move(Solution & /*_solution*/) {
	}
};

class moDummyBackableNeighbor: public moBackableNeighbor<Solution> {
public:
	virtual void move(Solution & /*_solution*/) {
	}
	virtual void moveBack(Solution & /*_solution*/) {
	}
};

class moDummyNeighborhoodTest: public moNeighborhood<moDummyNeighborTest> {
public:
	typedef moDummyNeighborTest Neighbor;

	moDummyNeighborhoodTest() :
		i(0), j(0) {
	}

	virtual bool hasNeighbor(EOT & /*_solution*/) {
		bool res;
		if (i % 3 == 0 || i == 1)
			res = false;
		else
			res = true;
		i++;
		return res;
	}
	virtual void init(EOT & /*_solution*/, Neighbor & /*_current*/) {
	}
	virtual void next(EOT & /*_solution*/, Neighbor & /*_current*/) {
	}
	virtual bool cont(EOT & /*_solution*/) {
		j++;
		return (j % 10 != 0);
	}

private:
	int i, j;
};

class moDummyEvalTest: public eoEvalFunc<Solution> {
public:
	void operator()(Solution& _sol) {
		if (_sol.invalid())
			_sol.fitness(100);
		else
			_sol.fitness(_sol.fitness() + 50);
	}
};

class evalOneMax: public moEval<bitNeighbor> {
private:
	unsigned size;

public:
	evalOneMax(unsigned _size) :
		size(_size) {
	}
	;

	~evalOneMax(void) {
	}
	;

	void operator()(bitVector& _sol, bitNeighbor& _n) {
		unsigned int fit = _sol.fitness();
		if (_sol[_n.index()])
			fit--;
		else
			fit++;
		_n.fitness(fit);
	}
};

class dummyEvalOneMax: public moEval<bitNeighbor> {
private:
	unsigned size;

public:
	dummyEvalOneMax(unsigned _size) :
		size(_size) {
	}
	;

	~dummyEvalOneMax(void) {
	}
	;

	void operator()(bitVector& _sol, bitNeighbor& _n) {
		unsigned int fit = _sol.fitness();
		_n.fitness(fit);
	}
};

class monitor1: public eoMonitor {
public:

	monitor1(unsigned int& _a) :
		a(_a) {
	}

	eoMonitor& operator()() {
		a++;
		return *this;
	}

	void lastCall() {
		a++;
	}

private:
	unsigned int& a;
};

class monitor2: public eoMonitor {
public:

	monitor2(unsigned int& _a) :
		a(_a) {
	}

	eoMonitor& operator()() {
		a++;
		return *this;
	}

	void lastCall() {
		a++;
	}

private:
	unsigned int& a;
};

class updater1: public eoUpdater {
public:
	updater1(unsigned int& _a) :
		a(_a) {
	}

	void operator()() {
		a++;
	}

	void lastCall() {
		a++;
	}

private:
	unsigned int& a;
};

class dummyInit: public eoInit<bitVector> {
public:
	void operator()(bitVector& /*sol*/) {
	}
};

class dummyInit2: public eoInit<bitVector> {

public:
	dummyInit2(unsigned int _size) :
		size(_size) {
	}

	void operator()(bitVector& sol) {
		sol.resize(0);
		for (unsigned int i = 0; i < size; i++)
			sol.push_back(true);
	}

private:
	unsigned int size;
};

#endif
