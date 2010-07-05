#ifndef _doEstimator_h
#define _doEstimator_h

#include <eoPop.h>
#include <eoFunctor.h>

template < typename D >
class doEstimator : public eoUF< eoPop< typename D::EOType >&, D >
{
public:
    typedef typename D::EOType EOType;

    // virtual D operator() ( eoPop< EOT >& )=0 (provided by eoUF< A1, R >)
};

#endif // !_doEstimator_h
