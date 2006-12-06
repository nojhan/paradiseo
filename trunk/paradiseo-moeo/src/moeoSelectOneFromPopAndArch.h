// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoSelectOneFormPopAndArch.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOSELECTONEFROMPOPANDARCH_H_
#define MOEOSELECTONEFROMPOPANDARCH_H_

#include <eoPop.h>
#include <eoRandomSelect.h>
#include <eoSelectOne.h>
#include <utils/eoRNG.h>
#include <moeoArchive.h>

/**
 * Elitist selection process that consists in choosing individuals in the archive as well as in the current population.
 */
template < class EOT > class moeoSelectOneFromPopAndArch:public eoSelectOne <
  EOT >
{
public:

	/**
	 * Ctor
	 * @param _popSelectOne the population's selection operator
	 * @param _archSelectOne the archive's selection operator
	 * @param _arch the archive
	 * @param _ratioFromPop the ratio of selected individuals from the population
	 */
moeoSelectOneFromPopAndArch (eoSelectOne < EOT > &_popSelectOne, eoSelectOne < EOT > _archSelectOne, moeoArchive < EOT > &_arch, double _ratioFromPop = 0.5):popSelectOne (_popSelectOne), archSelectOne (_archSelectOne), arch (_arch),
    ratioFromPop
    (_ratioFromPop)
  {
  }

	/**
	 * Ctor - the archive's selection operator is a random selector
	 * @param _popSelectOne the population's selection operator	 
	 * @param _arch the archive
	 * @param _ratioFromPop the ratio of selected individuals from the population
	 */
moeoSelectOneFromPopAndArch (eoSelectOne < EOT > &_popSelectOne, moeoArchive < EOT > &_arch, double _ratioFromPop = 0.5):popSelectOne (_popSelectOne), archSelectOne (randomSelect), arch (_arch),
    ratioFromPop
    (_ratioFromPop)
  {
  }

	/**
	 * The selection process
	 */
  virtual const EOT & operator   () (const eoPop < EOT > &pop)
  {
    if (arch.size () > 0)
      if (rng.flip (ratioFromPop))
	return popSelectOne (pop);
      else
	return archSelectOne (arch);
    else
      return popSelectOne (pop);
  }

	/**
	 * Setups some population stats
	 */
  virtual void setup (const eoPop < EOT > &_pop)
  {
    popSelectOne.setup (_pop);
  }


private:

	/** The population's selection operator */
  eoSelectOne < EOT > &popSelectOne;
	/** The archive's selection operator */
  eoSelectOne < EOT > &archSelectOne;
	/** the archive */
  moeoArchive < EOT > &arch;
	/** the ratio of selected individuals from the population*/
  double ratioFromPop;
	/** the random selection operator */
  eoRandomSelect < EOT > randomSelect;

};

#endif /*MOEOSELECTONEFROMPOPANDARCH_H_ */
