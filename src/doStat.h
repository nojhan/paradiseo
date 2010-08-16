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

template < typename D >
class doStat : doStatBase< D >
{
public:
    typedef typename D::EOType EOType;
    typedef typename EOType::AtomType AtomType;

public:
    doStat(){}


    D& distrib() { return _distrib; }

private:
    D& _distrib;
};

#endif // !_doStat_h
