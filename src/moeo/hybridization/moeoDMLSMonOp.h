/*
* <moeoDMLSMonOp.h>
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

#ifndef _MOEODMLSMONOP_H
#define _MOEODMLSMONOP_H

#include "../../eo/eoGenContinue.h"
#include "../../eo/utils/eoRNG.h"
#include "../../eo/eoEvalFunc.h"
#include "../archive/moeoArchive.h"
#include "../archive/moeoUnboundedArchive.h"
#include "../explorer/moeoPopNeighborhoodExplorer.h"
#include "../selection/moeoUnvisitedSelect.h"
#include "../algo/moeoUnifiedDominanceBasedLS.h"

/** eoMonOp allowing hybridization with a dmls at mutation */
template < class Neighbor >
class moeoDMLSMonOp : public eoMonOp < typename Neighbor::EOT >
{

	typedef typename Neighbor::EOT MOEOT;

	public :
	/** Ctor with a dmls.
	 * @param _dmls the dmls use for the hybridization (!!! Special care is needed when choosing the continuator of the dmls !!!)
	 * @param _dmlsArchive an archive (used to instantiate the dmls)
	 * @param _verbose verbose mode
	 */
    moeoDMLSMonOp(moeoUnifiedDominanceBasedLS <Neighbor> & _dmls,
    		moeoArchive < MOEOT > & _dmlsArchive,
    		bool _verbose = true):
    			defaultContinuator(0), dmlsArchive(_dmlsArchive), dmls(_dmls), verbose(_verbose)	{}

	/** Ctor with a dmls.
	 * @param _eval a evaluation function (used to instantiate the dmls)
	 * @param _explorer a neighborhood explorer (used to instantiate the dmls)
	 * @param _select a selector of unvisited individuals of a population (used to instantiate the dmls)
	 * @param _step (default=1) is the number of Generation of dmls (used to instantiate the defaultContinuator for the dmls)
	 * @param _verbose verbose mode
	 */
    moeoDMLSMonOp(eoEvalFunc < MOEOT > & _eval,
            moeoPopNeighborhoodExplorer < Neighbor > & _explorer,
            moeoUnvisitedSelect < MOEOT > & _select,
    		unsigned int _step=1,
    		bool _verbose = true):
    			defaultContinuator(_step), dmlsArchive(defaultArchive), dmls(defaultContinuator, _eval, defaultArchive, _explorer, _select), verbose(_verbose){}

    /** Ctor with a dmls.
	 * @param _eval a evaluation function (used to instantiate the dmls)
	 * @param _dmlsArchive an archive (used to instantiate the dmls)
	 * @param _explorer a neighborhood explorer (used to instantiate the dmls)
	 * @param _select a selector of unvisited individuals of a population (used to instantiate the dmls)
	 * @param _step (default=1) is the number of Generation of dmls (used to instantiate the defaultContinuator for the dmls)
	 * @param _verbose verbose mode
	 */
	moeoDMLSMonOp(eoEvalFunc < MOEOT > & _eval,
			moeoArchive < MOEOT > & _dmlsArchive,
			moeoPopNeighborhoodExplorer < Neighbor > & _explorer,
			moeoUnvisitedSelect < MOEOT > & _select,
			unsigned int _step=1,
			bool _verbose = true):
				defaultContinuator(_step), dmlsArchive(_dmlsArchive), dmls(defaultContinuator, _eval, _dmlsArchive, _explorer, _select), verbose(_verbose){}

  /** functor which allow to run the dmls on a MOEOT and return one of the resulting archive*/
    bool operator()( MOEOT & _moeo)
    {
    	if(verbose)
    		std::cout << std::endl << "moeoDMLSMonOp: dmls start" << std::endl;
    	unsigned int tmp;
		eoPop < MOEOT> pop;
		pop.push_back(_moeo);
    	dmls(pop);
		tmp = rng.random(dmlsArchive.size());
		_moeo = dmlsArchive[tmp];
		defaultContinuator.totalGenerations(defaultContinuator.totalGenerations());
    	if(verbose)
    		std::cout << "moeoDMLSMonOp: dmls stop" << std::endl << std::endl;
		return false;
    }

    /**
     * @return the class name
     */
  virtual std::string className(void) const { return "moeoDMLSMonOp"; }

private:
	/** defaultContinuator used for the dmls */
	eoGenContinue < MOEOT > defaultContinuator;
	/** dmls archive */
	moeoArchive < MOEOT > & dmlsArchive;
	/** default archive used for the dmls */
	moeoUnboundedArchive < MOEOT > defaultArchive;
	/** the dmls */
	moeoUnifiedDominanceBasedLS <Neighbor> dmls;
	/** verbose mode */
	bool verbose;
};

#endif /*_MOEODMLSMONOP_H_*/
