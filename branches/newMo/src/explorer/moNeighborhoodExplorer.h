#ifndef _neighborhoodExplorer_h
#define _neighborhoodExplorer_h

#include <neighborhood/moNeighborhood.h>
/*
  explore the neighborhood
*/
template< class NH >
class moNeighborhoodExplorer : public eoUF<typename NH::EOT & , void>
{
public:
    typedef NH Neighborhood ;
    typedef typename Neighborhood::EOT EOT ;

    // empty constructor
    moNeighborhoodExplorer() { } ;

    virtual void initParam (EOT & solution) = 0 ;

    virtual void updateParam (EOT & solution) = 0 ;

    virtual bool isContinue(EOT & solution) = 0 ;

    virtual void move(EOT & solution) = 0 ;

    virtual bool accept(EOT & solution) = 0 ;

    /** Return the class id.
    *  @return the class name as a std::string
    */
    virtual std::string className() const { return "moNeighborhoodExplorer"; }
};

#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
