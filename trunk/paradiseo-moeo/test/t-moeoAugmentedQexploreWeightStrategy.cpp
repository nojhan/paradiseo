/*
 * <t-moeoAugmentedQexploreWeightStrategy.cpp>
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2008
 *
 * Fraéncçois Legillon
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
// t-moeoAugmentedQexploreWeightStrategy.cpp
//-----------------------------------------------------------------------------

#include <eo>  // EO
#include <mo>  // MO
#include <moeo>

using std::cout;
using std::endl;

//-----------------------------------------------------------------------------
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
			return 3;
		}
};

typedef moeoRealObjectiveVector < ObjectiveVectorTraits > ObjectiveVector;

class Solution : public moeoRealVector < ObjectiveVector, double, double >
{
	public:
		Solution() : moeoRealVector < ObjectiveVector, double, double > (3) {}
};

class solutionEval : public eoEvalFunc < Solution >
{
	public:
		void operator () (Solution & _sol)
		{
			ObjectiveVector objVec;
			objVec[0] = _sol[0];
			objVec[1] = _sol[1];
			objVec[2] = _sol[2];
			_sol.objectiveVector(objVec);
		}
};


class testMove : public moMove <Solution>
{
	public :
		void operator () (Solution & _solution)
		{
			Solution sol=_solution;
		}
} ;

class testMoveInit : public moMoveInit <testMove>
{
	public :
		void operator () (testMove & _move, const Solution & _solution)
		{
			testMove move=_move;
			const Solution sol(_solution);
		}
} ;

class testMoveNext : public moNextMove <testMove>
{
	public :
		bool operator () (testMove & _move, const Solution & _solution)
		{
			testMove move=_move;
			const Solution sol(_solution);

			return false;
		}
} ;

class testMoveIncrEval : public moMoveIncrEval <testMove,ObjectiveVector>
{
	public :
		ObjectiveVector operator () (const testMove & _move, const Solution & _solution)
		{
			const testMove move(_move);
			const Solution solution(_solution);


			return _solution.objectiveVector();
		}
} ;

class testTabuList : public moTabuList<testMove>
{
	public:
		bool operator() (const testMove & _move, const Solution & _solution)
		{
			const testMove move(_move);
			const Solution sol(_solution);

			return false;
		}

		void add(const testMove & _move, const Solution & _solution)
		{
			const testMove move(_move);
			const Solution sol(_solution);
		}

		void update()
		{}

		void init()
		{}
};

class testAspirCrit : public moAspirCrit<testMove>
{
	public:
		bool operator() (const testMove & _move, const double & _fitness)
		{
			double fitness;
			const testMove move(_move);
			fitness=_fitness;

			return false;
		}

		void init()
		{}
};

class solutionContinue : public eoContinue<Solution>
{
	public :
		solutionContinue(): counter(0)
	{}

		bool operator () (const eoPop<Solution> & _solution)
		{
			if(counter==0)
			{
				counter++;
				return true;
			}
			return false;
		}


		bool operator () (const Solution & _solution)
		{
			const Solution sol(_solution);

			if(counter==0)
			{
				counter++;
				return true;
			}
			return false;
		}

		void init()
		{}
	private :
		unsigned int counter;
} ;

class solutionComparator : public moeoComparator<Solution>
{
	public :
		const bool operator () (const Solution & _solution1 , const Solution & _solution2)
		{
			const Solution sol1(_solution1);
			const Solution sol2(_solution2);

			return sol1.fitness()>sol2.fitness();
		}
} ;

class solutionPerturbation : public eoMonOp<Solution>
{
	public :
		bool operator () (Solution & _solution)
		{
			ObjectiveVector objVec;
			objVec[0] = 1;
			objVec[1] = 1;
			objVec[2] = 1;
			_solution[1]=1;
			_solution[0]=1;
			_solution[2]=1;
			_solution.objectiveVector(objVec);
			_solution.fitness(2);
			return true;
		}
} ;

class solutionSingler : public moeoSingleObjectivization<Solution>
{
	void operator () (Solution & _solution){
			ObjectiveVector objVec;
			objVec[0] = _solution[0];
			objVec[1] = _solution[1];
			objVec[2] = _solution[2];
			_solution.objectiveVector(objVec);
		_solution.fitness(_solution.objectiveVector()[0]+_solution.objectiveVector()[1]);
	}
	void operator()(eoPop<Solution> &_pop){
	}

	double operator() (const ObjectiveVector &_obj){
		return _obj[0]+_obj[1];
	}
	void updateByDeleting(eoPop<Solution>& pop, ObjectiveVector& obj){}

} ;

class selectMove: public moMoveSelect<testMove>
{
	public:
		void init(const double &d){
			max_fit=d;
		}
		bool update(const testMove &move,const double &fitness){
			if (fitness>max_fit){
				max_fit=fitness;
				best_move=move;
				return false;
			}else
				return true;
		}
		void operator()(testMove &move,double &fitness){
			move=best_move;
			fitness=max_fit;
		}
	private:
		double max_fit;
		testMove best_move;
};

//-----------------------------------------------------------------------------

int main()
{
	Solution solution;
	solutionEval eval;
	std::vector<double> weight;
	weight.resize(3);
	cout << "[moeoAugmentedQexploreWeightStrategy] ==> ";
	moeoAugmentedQexploreWeightStrategy<Solution> strat;
	for (unsigned int i=0;i<6000;i++){
		eval(solution);
		strat(weight,solution);
	//	std::cout<<weight[0]<<" "<<weight[1]<<" "<<weight[2]<<" "/*<<weight[3]*/<<std::endl;
	}

	std::cout<<"OK"<<std::endl;
	return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------

