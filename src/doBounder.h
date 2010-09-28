// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doBounder_h
#define _doBounder_h

#include <eoFunctor.h>

template < typename EOT >
class doBounder : public eoUF< EOT&, void >
{
public:
    doBounder( EOT min = EOT(1, 0), EOT max = EOT(1, 0) )
	: _min(min), _max(max)
    {
	assert(_min.size() > 0);
	assert(_min.size() == _max.size());
    }

    // virtual void operator()( EOT& ) = 0 (provided by eoUF< A1, R >)

    EOT& min(){return _min;}
    EOT& max(){return _max;}

private:
    EOT _min;
    EOT _max;
};

#endif // !_doBounder_h
