#include <eoGenContinue.h>
#include <utils/eoRNG.h>
#include <eoEvalFunc.h>
#include <archive/moeoArchive.h>
#include <archive/moeoUnboundedArchive.h>
#include <moeoPopNeighborhoodExplorer.h>
#include <moeoUnvisitedSelect.h>
#include <moeoUnifiedDominanceBasedLS.h>

/** eoMonOp allowing hybridization with a dmls at mutation */
template < class Move >
class moeoDMLSMonOp : public eoMonOp < typename Move::EOType >
{

	typedef typename Move::EOType MOEOT;

	public :
	/** Ctor with a dmls.
	 * @param _dmls the dmls use for the hybridization (!!! Special care is needed when choosing the continuator of the dmls !!!)
	 * @param _dmlsArchive an archive (used to instantiate the dmls)
	 * @param _verbose verbose mode
	 */
    moeoDMLSMonOp(moeoUnifiedDominanceBasedLS <Move> & _dmls,
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
            moeoPopNeighborhoodExplorer < Move > & _explorer,
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
			moeoPopNeighborhoodExplorer < Move > & _explorer,
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
	moeoUnifiedDominanceBasedLS <Move> dmls;
	/** verbose mode */
	bool verbose;
};
