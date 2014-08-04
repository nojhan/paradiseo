/*
* <moeoDMLSGenUpdater.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jérémie Humeau
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#ifndef _MOEODMLSGENUPDATER_H
#define _MOEODMLSGENUPDATER_H

#include "../../eo/eoGenContinue.h"
#include "../../eo/eoEvalFunc.h"
#include "../archive/moeoArchive.h"
#include "../archive/moeoUnboundedArchive.h"
#include "../explorer/moeoPopNeighborhoodExplorer.h"
#include "../selection/moeoUnvisitedSelect.h"
#include "../algo/moeoUnifiedDominanceBasedLS.h"

/** updater allowing hybridization with a dmls at checkpointing*/
template < class Neighbor >
class moeoDMLSGenUpdater : public eoUpdater
{

	typedef typename Neighbor::EOT MOEOT;

	public :
	/** Ctor with a dmls.
	 * @param _dmls the dmls use for the hybridization (!!! Special care is needed when choosing the continuator of the dmls !!!)
	 * @param _dmlsArchive an archive (used to instantiate the dmls)
	 * @param _globalArchive the same archive used in the other algorithm
	 * @param _continuator is a Generational Continuator which allow to run dmls on the global archive each X generation(s)
		 * @param _verbose verbose mode
	 */
    moeoDMLSGenUpdater(moeoUnifiedDominanceBasedLS <Neighbor> & _dmls,
    		moeoArchive < MOEOT > & _dmlsArchive,
    		moeoArchive < MOEOT > & _globalArchive,
    		eoGenContinue < MOEOT > & _continuator,
    		bool _verbose = true):
    			defaultContinuator(0), dmlsArchive(_dmlsArchive), dmls(_dmls), globalArchive(_globalArchive), continuator(_continuator), verbose(_verbose){}

	/** Ctor with a dmls.
	 * @param _eval a evaluation function (used to instantiate the dmls)
	 * @param _explorer a neighborhood explorer (used to instantiate the dmls)
	 * @param _select a selector of unvisited individuals of a population (used to instantiate the dmls)
	 * @param _globalArchive the same archive used in the other algorithm
	 * @param _continuator is a Generational Continuator which allow to run dmls on the global archive each X generation(s)
	 * @param _step (default=1) is the number of Generation of dmls (used to instantiate the defaultContinuator for the dmls)
	 * @param _verbose verbose mode
	 */
    moeoDMLSGenUpdater(eoEvalFunc < MOEOT > & _eval,
            moeoPopNeighborhoodExplorer < Neighbor > & _explorer,
            moeoUnvisitedSelect < MOEOT > & _select,
    		moeoArchive < MOEOT > & _globalArchive,
    		eoGenContinue < MOEOT > & _continuator,
    		unsigned int _step=1,
    		bool _verbose = true):
    			defaultContinuator(_step), dmlsArchive(defaultArchive), dmls(defaultContinuator, _eval, defaultArchive, _explorer, _select), globalArchive(_globalArchive), continuator(_continuator), verbose(_verbose){}

    /** Ctor with a dmls.
	 * @param _eval a evaluation function (used to instantiate the dmls)
	 * @param _dmlsArchive an archive (used to instantiate the dmls)
	 * @param _explorer a neighborhood explorer (used to instantiate the dmls)
	 * @param _select a selector of unvisited individuals of a population (used to instantiate the dmls)
	 * @param _globalArchive the same archive used in the other algorithm
	 * @param _continuator is a Generational Continuator which allow to run dmls on the global archive each X generation(s)
	 * @param _step (default=1) is the number of Generation of dmls (used to instantiate the defaultContinuator for the dmls)
	 * @param _verbose verbose mode
	 */
	moeoDMLSGenUpdater(eoEvalFunc < MOEOT > & _eval,
			moeoArchive < MOEOT > & _dmlsArchive,
			moeoPopNeighborhoodExplorer < Neighbor > & _explorer,
			moeoUnvisitedSelect < MOEOT > & _select,
			moeoArchive < MOEOT > & _globalArchive,
			eoGenContinue < MOEOT > & _continuator,
			unsigned int _step=1,
			bool _verbose = true):
				defaultContinuator(_step), dmlsArchive(_dmlsArchive), dmls(defaultContinuator, _eval, _dmlsArchive, _explorer, _select), globalArchive(_globalArchive), continuator(_continuator), verbose(_verbose){}

  /** functor which allow to run the dmls*/
    virtual void operator()()
    {
    	if(!continuator(globalArchive)){
    		if(verbose)
				std::cout << std::endl << "moeoDMLSGenUpdater: dmls start" << std::endl;
			dmls(globalArchive);
			globalArchive(dmlsArchive);
    		if(verbose)
				std::cout << "moeoDMLSGenUpdater: dmls stop" << std::endl;
			defaultContinuator.totalGenerations(defaultContinuator.totalGenerations());
    		if(verbose)
				std::cout << "the other algorithm  restart for " << continuator.totalGenerations() << " generation(s)" << std::endl << std::endl;
			continuator.totalGenerations(continuator.totalGenerations());
    	}
    }

    /**
     * @return the class name
     */
  virtual std::string className(void) const { return "moeoDMLSGenUpdater"; }

private:
	/** defaultContinuator used for the dmls */
	eoGenContinue < MOEOT > defaultContinuator;
	/** dmls archive */
	moeoArchive < MOEOT > & dmlsArchive;
	/** default archive used for the dmls */
	moeoUnboundedArchive < MOEOT > defaultArchive;
	/** the dmls */
	moeoUnifiedDominanceBasedLS <Neighbor> dmls;
	/** the global archive */
	moeoArchive < MOEOT > & globalArchive;
	/** continuator used to run the dmls each X generation(s) */
	eoGenContinue < MOEOT > & continuator;
	/** verbose mode */
	bool verbose;
};


#endif /*_MOEODMLSGENUPDATER_H_*/
