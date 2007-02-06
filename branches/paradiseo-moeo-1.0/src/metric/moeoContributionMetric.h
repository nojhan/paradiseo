// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoContributionMetric.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOCONTRIBUTIONMETRIC_H_
#define MOEOCONTRIBUTIONMETRIC_H_

#include <metric/moeoMetric.h>

/**
 * The contribution metric evaluates the proportion of non-dominated solutions given by a Pareto set relatively to another Pareto set
 * 
 * (Meunier, Talbi, Reininger: 'A multiobjective genetic algorithm for radio network optimization', in Proc. of the 2000 Congress on Evolutionary Computation, IEEE Press, pp. 317-324)
 */
template < class MOEOT >
class moeoContributionMetric : public moeoPopVsPopBinaryMetric < MOEOT, double >
{
public:
	
	/** the objective vector type of a solution */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	

	/**
	 * Returns the contribution of the Pareto set '_set1' relatively to the Pareto set '_set2'
	 * @param _set1 the first Pareto set
	 * @param _set2 the second Pareto set
	 */
	double operator()(const eoPop < MOEOT > & _pop1, const eoPop < MOEOT >  & _pop2) {
		/************/
		std::vector<ObjectiveVector> set1;
		std::vector<ObjectiveVector> set2;
		for (unsigned i=0; i<_pop1.size(); i++)
			set1.push_back(_pop1[i].objectiveVector());
		for (unsigned i=0 ; i<_pop2.size(); i++)
			set2.push_back(_pop2[i].objectiveVector());
		/****************/		
		unsigned c  = card_C(set1, set2);
		unsigned w1 = card_W(set1, set2);
		unsigned n1 = card_N(set1, set2);
		unsigned w2 = card_W(set2, set1);
		unsigned n2 = card_N(set2, set1);
		return (double) (c / 2.0 + w1 + n1) / (c + w1 + n1 + w2 + n2);
	}


private:

	/**
	 * Returns the number of solutions both in '_set1' and '_set2'
	 * @param _set1 the first Pareto set
	 * @param _set2 the second Pareto set
	 */
	unsigned card_C (const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2) {
		unsigned c=0;
		for (unsigned i=0; i<_set1.size(); i++)
			for (unsigned j=0; j<_set2.size(); j++)
				if (_set1[i] == _set2[j]) {
					c++;
					break;
				}
		return c;
	}

	/**
	 * Returns the number of solutions in '_set1' dominating at least one solution of '_set2'
	 * @param _set1 the first Pareto set
	 * @param _set2 the second Pareto set
	 */
	unsigned card_W (const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2) {
		unsigned w=0;
		for (unsigned i=0; i<_set1.size(); i++)
			for (unsigned j=0; j<_set2.size(); j++)
				if (_set1[i].dominates(_set2[j])) {
					w++;
					break;
				}
		return w;
	}	

	/**
	 * Returns the number of solutions in '_set1' having no relation of dominance with those from '_set2'
	 * @param _set1 the first Pareto set
	 * @param _set2 the second Pareto set
	 */
	unsigned card_N (const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2) {
		unsigned n=0;
		for (unsigned i=0; i<_set1.size(); i++) {
			bool domin_rel = false;
			for (unsigned j=0; j<_set2.size(); j++)
				if (_set1[i].dominates(_set2[j]) || _set2[j].dominates(_set1 [i])) {
					domin_rel = true;
					break;
				}
				if (! domin_rel)
					n++;
		}
		return n;
	}

};

#endif /*MOEOCONTRIBUTIONMETRIC_H_*/
