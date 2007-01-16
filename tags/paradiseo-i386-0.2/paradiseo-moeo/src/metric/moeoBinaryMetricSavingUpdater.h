// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoBinaryMetricSavingUpdater.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOBINARYMETRICSAVINGUPDATER_H_
#define MOEOBINARYMETRICSAVINGUPDATER_H_

#include <fstream>
#include <string>
#include <eoPop.h>
#include <utils/eoUpdater.h>
#include <metric/moeoMetric.h>

/** 
 * This class allows to save the progression of a binary metric comparing the fitness values of the current population (or archive) 
 * with the fitness values of the population (or archive) of the generation (n-1) into a file 
 */
template < class EOT > class moeoBinaryMetricSavingUpdater:public eoUpdater
{
public:

	/**
	 * The fitness type of a solution 
	 */
  typedef typename EOT::Fitness EOFitness;

	/**
	 * Ctor
	 * @param _metric the binary metric comparing two Pareto sets
	 * @param _pop the main population
	 * @param _filename the target filename
	 */
  moeoBinaryMetricSavingUpdater (moeoVectorVsVectorBM < EOT, double >&_metric,
				 const eoPop < EOT > &_pop,
				 std::string _filename):metric (_metric),
    pop (_pop), filename (_filename), counter (1)
  {
  }

	/**
	 * Saves the metric's value for the current generation
	 */
  void operator () ()
  {
    if (pop.size ())
      {
	if (firstGen)
	  {
	    firstGen = false;
	  }
	else
	  {
	    // creation of the two Pareto sets                              
	    std::vector < EOFitness > from;
	    std::vector < EOFitness > to;
	    for (unsigned i = 0; i < pop.size (); i++)
	      from.push_back (pop[i].fitness ());
	    for (unsigned i = 0; i < oldPop.size (); i++)
	      to.push_back (oldPop[i].fitness ());
	    // writing the result into the file
	    std::ofstream f (filename.c_str (), std::ios::app);
	    f << counter++ << ' ' << metric (from, to) << std::endl;
	    f.close ();
	  }
	oldPop = pop;
      }
  }

private:

	/** binary metric comparing two Pareto sets */
  moeoVectorVsVectorBM < EOT, double >&metric;
	/** main population */
  const eoPop < EOT > &pop;
	/** (n-1) population */
  eoPop < EOT > oldPop;
	/** target filename */
  std::string filename;
	/** is it the first generation ? */
  bool firstGen;
	/** counter */
  unsigned counter;

};

#endif /*MOEOBINARYMETRICSAVINGUPDATER_H_ */
