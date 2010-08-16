#ifndef _doCheckPoint_h
#define _doCheckPoint_h

#include "doContinue.h"
#include "doStat.h"

//! eoCheckPoint< EOT > classe fitted to Distribution Object library

template < typename D >
class doCheckPoint : public doContinue< D >
{
public:
    typedef typename D::EOType EOType;

    doCheckPoint(doContinue< D >& _cont)
    {
	_continuators.push_back( &_cont );
    }

    bool operator()(const D& distrib)
    {
	for ( unsigned int i = 0, size = _stats.size(); i < size; ++i )
	    {
		(*_stats[i])( distrib );
	    }

	bool bContinue = true;
	for ( unsigned int i = 0, size = _continuators.size(); i < size; ++i )
	    {
		if ( !(*_continuators[i]( distrib )) )
		    {
			bContinue = false;
		    }
	    }

	if ( !bContinue )
	    {
		for ( unsigned int i = 0, size = _stats.size(); i < size; ++i )
		    {
			_stats[i]->lastCall( distrib );
		    }
	    }

	return bContinue;
    }

    void add(doContinue< D >& cont) { _continuators.push_back( &cont ); }
    void add(doStatBase< D >& stat) { _stats.push_back( &stat ); }

    virtual std::string className(void) const { return "doCheckPoint"; }

    std::string allClassNames() const
    {
	std::string s("\n" + className() + "\n");

	s += "Stats\n";
	for ( unsigned int i = 0, size = _stats.size(); i < size; ++i )
	    {
		s += _stats[i]->className() + "\n";
	    }
	s += "\n";

	s += "Continuators\n";
	for ( unsigned int i = 0, size = _continuators.size(); i < size; ++i )
	    {
		s += _continuators[i]->className() + "\n";
	    }
	s += "\n";

	return s;
    }

private:
    std::vector< doContinue< D >* > _continuators;
    std::vector< doStatBase< D >* > _stats;
};

#endif // !_doCheckPoint_h
