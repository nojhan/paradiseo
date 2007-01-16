// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoCombinedMOLS.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOCOMBINEDMOLS_H_
#define MOEOCOMBINEDMOLS_H_

#include <eoEvalFunc.h>
#include <moeoArchive.h>
#include <moeoMOLS.h>

/**
 * This class allows to embed a set of local searches that are sequentially applied, 
 * and so working and updating the same archive of non-dominated solutions
 */
template < class EOT > class moeoCombinedMOLS:public moeoMOLS < EOT >
{
public:

	/**
	 * Ctor
	 * @param _eval the full evaluator of a solution
	 * @param _first_ls the first multi-objective local search to add
	 */
moeoCombinedMOLS (eoEvalFunc < EOT > &_eval, moeoMOLS < EOT > &_first_ls):eval
    (_eval)
  {
    combinedMOLS.push_back (&_first_ls);
  }

	/**
	 * Adds a new local search to combine
	 * @param _ls the multi-objective local search to add
	 */
  void add (moeoMOLS < EOT > &_ls)
  {
    combinedMOLS.push_back (&_ls);
  }

	/**
	 * Gives a new solution in order to explore the neigborhood. 
	 * The new non-dominated solutions are added to the archive
	 * @param _eo the solution
	 * @param _arch the archive of non-dominated solutions 
	 */
  void operator  () (const EOT & _eo, moeoArchive < EOT > &_arch)
  {
    eval (const_cast < EOT & >(_eo));
    for (unsigned i = 0; i < combinedMOLS.size (); i++)
      combinedMOLS[i]->operator ()(_eo, _arch);
  }


private:

	/** the full evaluator of a solution */
  eoEvalFunc < EOT > &eval;
	/** the vector that contains the combined MOLS */
  std::vector < moeoMOLS < EOT > *>combinedMOLS;

};

#endif /*MOEOCOMBINEDMOLS_H_ */
