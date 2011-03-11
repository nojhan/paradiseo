#include <eoGenContinue.h>
#include <eoEvalFunc.h>
#include <archive/moeoArchive.h>
#include <archive/moeoUnboundedArchive.h>
#include <moeoPopNeighborhoodExplorer.h>
#include <moeoUnvisitedSelect.h>
#include <moeoUnifiedDominanceBasedLS.h>

/** updater allowing hybridization with a dmls at checkpointing*/
template < class Move >
class moeoDMLSGenUpdater : public eoUpdater
{

	typedef typename Move::EOType MOEOT;

	public :
	/** Ctor with a dmls.
	 * @param _dmls the dmls use for the hybridization (!!! Special care is needed when choosing the continuator of the dmls !!!)
	 * @param _dmlsArchive an archive (used to instantiate the dmls)
	 * @param _globalArchive the same archive used in the other algorithm
	 * @param _continuator is a Generational Continuator which allow to run dmls on the global archive each X generation(s)
		 * @param _verbose verbose mode
	 */
    moeoDMLSGenUpdater(moeoUnifiedDominanceBasedLS <Move> & _dmls,
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
            moeoPopNeighborhoodExplorer < Move > & _explorer,
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
			moeoPopNeighborhoodExplorer < Move > & _explorer,
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
	moeoUnifiedDominanceBasedLS <Move> dmls;
	/** the global archive */
	moeoArchive < MOEOT > & globalArchive;
	/** continuator used to run the dmls each X generation(s) */
	eoGenContinue < MOEOT > & continuator;
	/** verbose mode */
	bool verbose;
};
