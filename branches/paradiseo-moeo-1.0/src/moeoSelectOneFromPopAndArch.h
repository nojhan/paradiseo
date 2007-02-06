// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoSelectOneFormPopAndArch.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOSELECTONEFROMPOPANDARCH_H_
#define MOEOSELECTONEFROMPOPANDARCH_H_

#include <eoPop.h>
#include <utils/eoRNG.h>
#include <moeoArchive.h>
#include <moeoSelectOne.h>
#include <moeoRandomSelectOne.h>

/**
 * Elitist selection process that consists in choosing individuals in the archive as well as in the current population.
 */
template<class EOT>
class moeoSelectOneFromPopAndArch : public moeoSelectOne<EOT>
{
public:

	/**
	 * Ctor
	 * @param _popSelectOne the population's selection operator
	 * @param _archSelectOne the archive's selection operator
	 * @param _arch the archive
	 * @param _ratioFromPop the ratio of selected individuals from the population
	 */
	moeoSelectOneFromPopAndArch (moeoSelectOne<EOT> & _popSelectOne, moeoSelectOne<EOT> _archSelectOne, moeoArchive <EOT> & _arch, double _ratioFromPop=0.5)
	 : popSelectOne(_popSelectOne), archSelectOne(_archSelectOne), arch(_arch), ratioFromPop(_ratioFromPop)
	{}
	
	/**
	 * Ctor - the archive's selection operator is a random selector
	 * @param _popSelectOne the population's selection operator	 
	 * @param _arch the archive
	 * @param _ratioFromPop the ratio of selected individuals from the population
	 */
	moeoSelectOneFromPopAndArch (moeoSelectOne<EOT> & _popSelectOne, moeoArchive <EOT> & _arch, double _ratioFromPop=0.5)
	 : popSelectOne(_popSelectOne), archSelectOne(randomSelectOne), arch(_arch), ratioFromPop(_ratioFromPop)
	{}	
	
	/**
	 * The selection process
	 */
	virtual const EOT & operator () (const eoPop<EOT> & pop) {
		if (arch.size() > 0)
			if (rng.flip(ratioFromPop))
				return popSelectOne(pop);
			else
				return archSelectOne(arch);
		else
			return popSelectOne(pop);
	}
	
	/**
	 * Setups some population stats
	 */
	virtual void setup (const eoPop<EOT> & _pop) {
		popSelectOne.setup(_pop);
	}


private:

	/** The population's selection operator */
	moeoSelectOne<EOT> & popSelectOne;
	/** The archive's selection operator */
	moeoSelectOne<EOT> & archSelectOne;
	/** The archive */
	moeoArchive <EOT> & arch;
	/** The ratio of selected individuals from the population*/
	double ratioFromPop;
	/** A random selection operator */
	moeoRandomSelectOne<EOT> randomSelectOne;
	
};

#endif /*MOEOSELECTONEFROMPOPANDARCH_H_*/
