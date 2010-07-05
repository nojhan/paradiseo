#ifndef _doBounderNo_h
#define _doBounderNo_h

#include "doBounder.h"

template < typename EOT >
class doBounderNo : public doBounder< EOT >
{
public:
    void operator()( EOT& x )
    {}
};

#endif // !_doBounderNo_h
