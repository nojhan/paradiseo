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
template < class EOT > class moeoContributionMetric:public moeoVectorVsVectorBM < EOT,
  double >
{
public:

	/**
	 * The fitness type of a solution 
	 */
  typedef typename EOT::Fitness EOFitness;

	/**
	 * Returns the contribution of the Pareto set '_set1' relatively to the Pareto set '_set2'
	 * @param _set1 the first Pareto set
	 * @param _set2 the second Pareto set
	 */
  double operator () (const std::vector < EOFitness > &_set1,
		      const std::vector < EOFitness > &_set2)
  {
    unsigned c = card_C (_set1, _set2);
    unsigned w1 = card_W (_set1, _set2);
    unsigned n1 = card_N (_set1, _set2);
    unsigned w2 = card_W (_set2, _set1);
    unsigned n2 = card_N (_set2, _set1);
      return (double) (c / 2.0 + w1 + n1) / (c + w1 + n1 + w2 + n2);
  }


private:

	/**
	 * Returns the number of solutions both in '_set1' and '_set2'
	 * @param _set1 the first Pareto set
	 * @param _set2 the second Pareto set
	 */
  unsigned card_C (const std::vector < EOFitness > &_set1,
		   const std::vector < EOFitness > &_set2)
  {
    unsigned c = 0;
    for (unsigned i = 0; i < _set1.size (); i++)
      for (unsigned j = 0; j < _set2.size (); j++)
	if (_set1[i] == _set2[j])
	  {
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
  unsigned card_W (const std::vector < EOFitness > &_set1,
		   const std::vector < EOFitness > &_set2)
  {
    unsigned w = 0;
    for (unsigned i = 0; i < _set1.size (); i++)
      for (unsigned j = 0; j < _set2.size (); j++)
	if (_set1[i].dominates (_set2[j]))
	  {
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
  unsigned card_N (const std::vector < EOFitness > &_set1,
		   const std::vector < EOFitness > &_set2)
  {
    unsigned n = 0;
    for (unsigned i = 0; i < _set1.size (); i++)
      {
	bool domin_rel = false;
	for (unsigned j = 0; j < _set2.size (); j++)
	  if (_set1[i].dominates (_set2[j]) || _set2[j].dominates (_set1[i]))
	    {
	      domin_rel = true;
	      break;
	    }
	if (!domin_rel)
	  n++;
      }
    return n;
  }

};

#endif /*MOEOCONTRIBUTIONMETRIC_H_ */
