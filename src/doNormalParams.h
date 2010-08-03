#ifndef _doNormalParams_h
#define _doNormalParams_h

#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/lu.hpp>

namespace ublas = boost::numeric::ublas;

template < typename EOT >
class doNormalParams
{
public:
    typedef typename EOT::AtomType AtomType;

    doNormalParams(const EOT& mean, const ublas::symmetric_matrix< AtomType, ublas::lower >& varcovar)
	: _mean(mean), _varcovar(varcovar)
    {
	assert(_mean.size() > 0);
	assert(_mean.size() == _varcovar.size1());
	assert(_mean.size() == _varcovar.size2());
    }

    EOT& mean(){return _mean;}
    ublas::symmetric_matrix< AtomType, ublas::lower >& varcovar(){return _varcovar;}

    unsigned int size()
    {
	assert(_mean.size() == _varcovar.size1());
	assert(_mean.size() == _varcovar.size2());
	return _mean.size();
    }

private:
    EOT _mean;
    ublas::symmetric_matrix< AtomType, ublas::lower > _varcovar;
};

#endif // !_doNormalParams_h
