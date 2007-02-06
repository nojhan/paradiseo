// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoArchiveUpdater.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOARCHIVEUPDATER_H_
#define MOEOARCHIVEUPDATER_H_

#include <eoPop.h>
#include <utils/eoUpdater.h>
#include <moeoArchive.h>

/**
 * This class allows to update the archive at each generation with newly found non-dominated solutions
 */
template < class EOT >
class moeoArchiveUpdater : public eoUpdater
{
public:
  
  /**
   * Ctor
   * @param _arch an archive of non-dominated solutions
   * @param _pop the main population
   */
  moeoArchiveUpdater(moeoArchive <EOT> & _arch, const eoPop<EOT> & _pop) : arch(_arch), pop(_pop)
  {}
  
  
  /**
   * Updates the archive with newly found non-dominated solutions contained in the main population
   */
  void operator()() {    
    arch.update(pop);    
  }


private:

	/** the archive of non-dominated solutions */
	moeoArchive<EOT> & arch;
	/** the main population */
	const eoPop<EOT> & pop;

};

#endif /*MOEOARCHIVEUPDATER_H_*/
