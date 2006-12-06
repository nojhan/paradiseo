// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoArchive.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOARCHIVE_H_
#define MOEOARCHIVE_H_

#include <eoPop.h>

/**
 * An archive is a secondary population that stores non-dominated solutions
 */
template < class EOT > class moeoArchive:public eoPop < EOT >
{
public:

  using std::vector < EOT >::size;
  using std::vector < EOT >::operator[];
  using std::vector < EOT >::back;
  using std::vector < EOT >::pop_back;

	/**
	 * The fitness type of a solution 
	 */
  typedef typename EOT::Fitness EOFitness;

  /**
   * Returns true if the current archive dominates _fit
   * @param _fit the (Pareto) fitness to compare with the current archive
   */
  bool dominates (const EOFitness & _fit) const
  {
    for (unsigned i = 0; i < size; i++)
      if (operator[](i).fitness ().dominates (_fit))
	return true;
    return false;
  }

  /**
   * Returns true if the current archive contains _fit
   * @param _fit the (Pareto) fitness to search within the current archive
   */
  bool contains (const EOFitness & _fit) const
  {
    for (unsigned i = 0; i < size; i++)
      if (operator[](i).fitness () == _fit)
	return true;
    return false;
  }

  /**
   * Updates the archive with a given individual _eo
   * @param _eo the given individual
   */
  void update (const EOT & _eo)
  {
    // Removing the dominated solutions from the archive
    for (unsigned j = 0; j < size ();)
      {
	if (_eo.fitness ().dominates (operator[](j).fitness ()))
	  {
	    operator[](j) = back ();
	    pop_back ();
	  }
	else if (_eo.fitness () == operator[](j).fitness ())
	  {
	    operator[](j) = back ();
	    pop_back ();
	  }
	else
	  j++;
      }

    // Dominated ?
    bool dom = false;
    for (unsigned j = 0; j < size (); j++)
      if (operator [](j).fitness ().dominates (_eo.fitness ()))
	{
	  dom = true;
	  break;
	}
    if (!dom)
      push_back (_eo);
  }

  /**
   * Updates the archive with a given population _pop
   * @param _pop the given population
   */
  void update (const eoPop < EOT > &_pop)
  {
    for (unsigned i = 0; i < _pop.size (); i++)
      update (_pop[i]);
  }

};

#endif /*MOEOARCHIVE_H_ */
