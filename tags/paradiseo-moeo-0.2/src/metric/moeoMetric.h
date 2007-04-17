// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoMetric.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOMETRIC_H_
#define MOEOMETRIC_H_

#include <eoFunctor.h>

/**
 * Base class for performance metrics (also called quality indicators)
 */
class moeoMetric:public eoFunctorBase
{
};


/**
 * Base class for unary metrics
 */
template < class A, class R > class moeoUM:public eoUF < A, R >,
  public moeoMetric
{
};


/**
 * Base class for binary metrics
 */
template < class A1, class A2, class R > class moeoBM:public eoBF < A1, A2, R >,
  public moeoMetric
{
};


/**
 * Base class for unary metrics dedicated to the performance evaluation of a single solution's Pareto fitness
 */
template < class EOT, class R, class EOFitness = typename EOT::Fitness > class moeoSolutionUM:public moeoUM <
  const
  EOFitness &,
  R >
{
};


/**
 * Base class for unary metrics dedicated to the performance evaluation of a Pareto set (a vector of Pareto fitnesses)
 */
template < class EOT, class R, class EOFitness = typename EOT::Fitness > class moeoVectorUM:public moeoUM <
  const
  std::vector <
EOFitness > &,
  R >
{
};


/**
 * Base class for binary metrics dedicated to the performance comparison between two solutions's Pareto fitnesses
 */
template < class EOT, class R, class EOFitness = typename EOT::Fitness > class moeoSolutionVsSolutionBM:public moeoBM <
  const
  EOFitness &, const
  EOFitness &,
  R >
{
};


/**
 * Base class for binary metrics dedicated to the performance comparison between a Pareto set (a vector of Pareto fitnesses) and a single solution's Pareto fitness
 */
template < class EOT, class R, class EOFitness = typename EOT::Fitness > class moeoVectorVsSolutionBM:public moeoBM <
  const
  std::vector <
EOFitness > &, const
  EOFitness &,
  R >
{
};


/**
 * Base class for binary metrics dedicated to the performance comparison between two Pareto sets (two vectors of Pareto fitnesses)
 */
template < class EOT, class R, class EOFitness = typename EOT::Fitness > class moeoVectorVsVectorBM:public moeoBM <
  const
  std::vector <
EOFitness > &, const
  std::vector <
EOFitness > &,
  R >
{
};


#endif /*MOEOMETRIC_H_ */
