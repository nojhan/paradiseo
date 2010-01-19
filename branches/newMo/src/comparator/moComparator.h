#ifndef _moComparator_h
#define _moComparator_h

#include <EO.h>
#include <eoFunctor.h>

// moComparator => comparer deux solutions
// idée :
// - eoComparator
// - moComparator qui hérite de eoComparator ?
// - moeoComparator qui hérite de eoComparator
// idée J :
// - eoComparator<TYPE> : eoBF <const TYPE & , const TYPE & , bool>
// - eoSolComparator : eoComparator<EOT> ?
// - moNeighborCompartor : : eoComparator<Neighbor>
//
// une instantiation possible !!
template< class EOT >
class moComparator : public eoBF<const EOT & , const EOT & , bool>
{
public:

    /*
     * Compare two solutions
     * @param _sol1 the first solution
     * @param _sol2 the second solution
     * @return true if the _sol1 is better than _sol2
     */
    virtual bool operator()(const EOT& _sol1, const EOT& _sol2) {
    	return (_sol1.fitness() > _sol2.fitness());
    }

    /*
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const {
    	return "moComparator";
    }
};

#endif

