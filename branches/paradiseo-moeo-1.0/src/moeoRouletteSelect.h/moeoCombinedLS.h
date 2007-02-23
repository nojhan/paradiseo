// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoCombinedLS.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOCOMBINEDLS_H_
#define MOEOCOMBINEDLS_H_

#include <moeoArchive.h>
#include <moeoEvalFunc.h>
#include <moeoLS.h>

/**
 * This class allows to embed a set of local searches that are sequentially applied, 
 * and so working and updating the same archive of non-dominated solutions
 */
template < class MOEOT > class moeoCombinedLS:public moeoLS < MOEOT >
{
public:

	/**
	 * Ctor
	 * @param _eval the full evaluator of a solution
	 * @param _first_ls the first multi-objective local search to add
	 */
moeoCombinedLS (moeoEvalFunc < MOEOT > &_eval, moeoLS < MOEOT > &_first_ls):eval
    (_eval)
  {
    combinedLS.push_back (&_first_ls);
  }

	/**
	 * Adds a new local search to combine
	 * @param _ls the multi-objective local search to add
	 */
  void add (moeoLS < MOEOT > &_ls)
  {
    combinedMOLS.push_back (&_ls);
  }

	/**
	 * Gives a new solution in order to explore the neigborhood. 
	 * The new non-dominated solutions are added to the archive
	 * @param _eo the solution
	 * @param _arch the archive of non-dominated solutions 
	 */
  void operator  () (const MOEOT & _eo, moeoArchive < MOEOT > &_arch)
  {
    eval (const_cast < MOEOT & >(_eo));
    for (unsigned i = 0; i < combinedLS.size (); i++)
      combinedLS[i]->operator ()(_eo, _arch);
  }


private:

	/** the full evaluator of a solution */
  moeoEvalFunc < MOEOT > &eval;
	/** the vector that contains the combined LS */
  std::vector < moeoLS < MOEOT > *>combinedLS;

};

#endif /*MOEOCOMBINEDLS_H_ */
