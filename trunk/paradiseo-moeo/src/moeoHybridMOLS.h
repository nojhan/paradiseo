// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoHybridMOLS.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOHYBRIDMOLS_H_
#define MOEOHYBRIDMOLS_H_

#include <eoContinue.h>
#include <eoPop.h>
#include <eoUpdater.h>
#include <eoSelect.h>
#include <moeoArchive.h>
#include <moeoMOLS.h>

/**
 * This class allows to apply a multi-objective local search to a number of selected individuals contained in the archive  
 * at every generation until a stopping criteria is verified.
 */
template < class EOT > class moeoHybridMOLS:public eoUpdater
{
public:

	/**
	 * Ctor
	 * @param _term stopping criteria
	 * @param _select selector
	 * @param _mols a multi-objective local search
	 * @param _arch the archive
	 */
moeoHybridMOLS (eoContinue < EOT > &_term, eoSelect < EOT > &_select, moeoMOLS < EOT > &_mols, moeoArchive < EOT > &_arch):term (_term), select (_select), mols (_mols),
    arch
    (_arch)
  {
  }

	/**
	 * Applies the multi-objective local search to selected individuals contained in the archive if the stopping criteria is not verified 
	 */
  void operator  () ()
  {
    if (!term (arch))
      {
	// selection of solutions
	eoPop < EOT > selectedSolutions;
	select (arch, selectedSolutions);
	// apply the local search to every selected solution
	for (unsigned i = 0; i < selectedSolutions.size (); i++)
	  mols (selectedSolutions[i], arch);
      }
  }


private:

	/** stopping criteria*/
  eoContinue < EOT > &term;
	/** selector */
  eoSelect < EOT > &select;
	/** multi-objective local search */
  moeoMOLS < EOT > &mols;
	/** archive */
  moeoArchive < EOT > &arch;

};

#endif /*MOEOHYBRIDMOLS_H_ */
