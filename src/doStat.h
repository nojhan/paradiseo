#ifndef _doStat_h
#define _doStat_h

#include <eoFunctor.h>

template < typename D >
class doStatBase : public eoUF< const D&, void >
{
public:
    // virtual void operator()( const D& ) = 0 (provided by eoUF< A1, R >)

    virtual void lastCall( const D& ) {}
    virtual std::string className() const { return "doStatBase"; }
};

template < typename D > class doCheckPoint;

template < typename D, typename T >
class doStat : public eoValueParam< T >, public doStatBase< D >
{
public:
    doStat(T value, std::string description)
	: eoValueParam< T >(value, description)
    {}

    virtual std::string className(void) const { return "doStat"; }

    doStat< D, T >& addTo(doCheckPoint< D >& cp) { cp.add(*this); return *this; }

    // TODO: doStat< D, T >& addTo(doMonitor& mon) { mon.add(*this); return *this; }
};


//! A parent class for any kind of distribution to dump parameter to std::string type

template < typename D >
class doDistribStat : public doStat< D, std::string >
{
public:
    using doStat< D, std::string >::value;

    doDistribStat(std::string desc)
	: doStat< D, std::string >("", desc)
    {}
};

#endif // !_doStat_h
