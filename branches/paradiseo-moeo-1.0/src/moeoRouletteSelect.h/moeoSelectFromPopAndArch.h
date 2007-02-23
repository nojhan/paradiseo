// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoSelectFormPopAndArch.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOSELECTFROMPOPANDARCH_H_
#define MOEOSELECTFROMPOPANDARCH_H_

#include <eoPop.h>
#include <utils/eoRNG.h>
#include <moeoArchive.h>
#include <moeoSelectOne.h>
#include <moeoRandomSelect.h>

/**
 * Elitist selection process that consists in choosing individuals in the archive as well as in the current population.
 */
template < class EOT > class moeoSelectFromPopAndArch:public moeoSelectOne <
  EOT >
{
public:

	/**
	 * Ctor
	 * @param _popSelect the population's selection operator
	 * @param _archSelect the archive's selection operator
	 * @param _arch the archive
	 * @param _ratioFromPop the ratio of selected individuals from the population
	 */
moeoSelectFromPopAndArch (moeoSelectOne < EOT > &_popSelect, moeoSelectOne < EOT > &_archSelect, moeoArchive < EOT > &_arch, double _ratioFromPop = 0.5):popSelect (_popSelect), archSelect (_archSelect), arch (_arch),
    ratioFromPop
    (_ratioFromPop)
  {
  }

	/**
	 * Ctor - the archive's selection operator is a random selector
	 * @param _popSelect the population's selection operator	 
	 * @param _arch the archive
	 * @param _ratioFromPop the ratio of selected individuals from the population
	 */
moeoSelectFromPopAndArch (moeoSelectOne < EOT > &_popSelect, moeoArchive < EOT > &_arch, double _ratioFromPop = 0.5):popSelect (_popSelect), archSelect (randomSelect), arch (_arch),
    ratioFromPop
    (_ratioFromPop)
  {
  }

	/**
	 * The selection process
	 */
  virtual const EOT & operator  () (const eoPop < EOT > &pop)
  {
    if (arch.size () > 0)
      if (rng.flip (ratioFromPop))
	return popSelect (pop);
      else
	return archSelect (arch);
    else
      return popSelect (pop);
  }

	/**
	 * Setups some population stats
	 */
  virtual void setup (const eoPop < EOT > &_pop)
  {
    popSelect.setup (_pop);
  }


private:

	/** The population's selection operator */
  moeoSelectOne < EOT > &popSelect;
	/** The archive's selection operator */
  moeoSelectOne < EOT > &archSelect;
	/** The archive */
  moeoArchive < EOT > &arch;
	/** The ratio of selected individuals from the population*/
  double ratioFromPop;
	/** A random selection operator */
  moeoRandomSelect < EOT > randomSelect;

};

#endif /*MOEOSELECTFROMPOPANDARCH_H_ */
