// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoReplacement.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOREPLACEMENT_H_
#define MOEOREPLACEMENT_H_

#include <eoPerf2Worth.h>
#include <eoPop.h>
#include <eoReplacement.h>


/**
 * Replacement strategy for multi-objective optimization
 */
template < class EOT, class WorthT > class moeoReplacement:public eoReplacement <
  EOT >
{
};


/**
 * Keep all the best individuals
 * (almost cut-and-pasted from eoNDPlusReplacement, (c) Maarten Keijzer, Marc Schoenauer and GeNeura Team, 2002)
 */
template < class EOT, class WorthT =
  double >class moeoElitistReplacement:public moeoReplacement < EOT, WorthT >
{
public:

  /**
   * constructor
   * @param _perf2worth the functor class to transform raw fitnesses into fitness for selection
   */
  moeoElitistReplacement (eoPerf2Worth < EOT,
			  WorthT > &_perf2worth):perf2worth (_perf2worth)
  {
  }


  /**
   * replacement - result in _parents
   * @param _parents parents population
   * @param _offspring offspring population
   */
  void operator () (eoPop < EOT > &_parents, eoPop < EOT > &_offspring)
  {
    unsigned size = _parents.size ();
    _parents.reserve (_parents.size () + _offspring.size ());
    copy (_offspring.begin (), _offspring.end (), back_inserter (_parents));

    // calculate worths
    perf2worth (_parents);
    perf2worth.sort_pop (_parents);
    perf2worth.resize (_parents, size);

    _offspring.clear ();
  }

private:
  /** the functor object to transform raw fitnesses into fitness for selection */
  eoPerf2Worth < EOT, WorthT > &perf2worth;
};


/**
 * Same than moeoElitistReplacement except that distinct individuals are privilegied
 */
template < class EOT, class WorthT =
  double >class moeoDisctinctElitistReplacement:public moeoReplacement < EOT,
  WorthT >
{
public:

  /**
   * constructor
   * @param _perf2worth the functor class to transform raw fitnesses into fitness for selection
   */
  moeoDisctinctElitistReplacement (eoPerf2Worth < EOT,
				   WorthT >
				   &_perf2worth):perf2worth (_perf2worth)
  {
  }


  /**
   * replacement - result in _parents
   * @param _parents parents population
   * @param _offspring offspring population
   */
  void operator () (eoPop < EOT > &_parents, eoPop < EOT > &_offspring)
  {
    unsigned size = _parents.size ();
    _parents.reserve (_parents.size () + _offspring.size ());
    copy (_offspring.begin (), _offspring.end (), back_inserter (_parents));

    // creation of the new population (of size 'size')
    createNewPop (_parents, size);

    _offspring.clear ();
  }


private:

  /** the functor object to transform raw fitnesses into fitness for selection */
  eoPerf2Worth < EOT, WorthT > &perf2worth;


  /**
   * creation of the new population of size _size
   * @param _pop the initial population (will be modified)
   * @param _size the size of the population to create
   */
  void createNewPop (eoPop < EOT > &_pop, unsigned _size)
  {
    // the number of occurences for each individual
    std::map < EOT, unsigned >nb_occurences;
    for (unsigned i = 0; i < _pop.size (); i++)
      nb_occurences[_pop[i]] = 0;
    // the new population
    eoPop < EOT > new_pop;
    new_pop.reserve (_pop.size ());
    for (unsigned i = 0; i < _pop.size (); i++)
      {
	if (nb_occurences[_pop[i]] == 0)
	  new_pop.push_back (_pop[i]);
	nb_occurences[_pop[i]]++;
      }

    // calculate worths (on the new population)
    perf2worth (new_pop);
    perf2worth.sort_pop (new_pop);

    // if case there's not enough individuals in the population...
    unsigned new_pop_size_init = new_pop.size ();
    unsigned k = 0;
    while (new_pop.size () < _size)
      {
	if (k < new_pop_size_init)
	  {
	    if (nb_occurences[new_pop[k]] > 1)
	      {
		new_pop.push_back (new_pop[k]);
		nb_occurences[new_pop[k]]--;
	      }
	    k++;
	  }
	else
	  k = 0;
      }

    // resize and swap the populations
    perf2worth.resize (new_pop, _size);
    _pop.resize (_size);
    _pop.swap (new_pop);
  }

};

#endif /*MOEOREPLACEMENT_H_ */
