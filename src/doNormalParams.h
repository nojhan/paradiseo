#ifndef _doNormalParams_h
#define _doNormalParams_h

<<<<<<< HEAD
=======
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/lu.hpp>

namespace ublas = boost::numeric::ublas;

>>>>>>> 36ec42d36204631eb4c25ae7b31a8728903697f8
template < typename EOT >
class doNormalParams
{
public:
<<<<<<< HEAD
    doNormalParams(EOT mean, EOT variance)
	: _mean(mean), _variance(variance)
    {
	assert(_mean.size() > 0);
	assert(_mean.size() == _variance.size());
    }

    EOT& mean(){return _mean;}
    EOT& variance(){return _variance;}

    unsigned int size()
    {
	assert(_mean.size() == _variance.size());
=======
    typedef typename EOT::AtomType AtomType;

    doNormalParams
    (
     const ublas::vector< AtomType >& mean,
     const ublas::symmetric_matrix< AtomType, ublas::lower >& varcovar
     )
	: _mean(mean), _varcovar(varcovar)
    {
	assert(_mean.size() > 0);
	assert(_mean.size() == _varcovar.size1());
	assert(_mean.size() == _varcovar.size2());
    }

    ublas::vector< AtomType >& mean(){return _mean;}
    ublas::symmetric_matrix< AtomType, ublas::lower >& varcovar(){return _varcovar;}

    unsigned int size()
    {
	assert(_mean.size() == _varcovar.size1());
	assert(_mean.size() == _varcovar.size2());
>>>>>>> 36ec42d36204631eb4c25ae7b31a8728903697f8
	return _mean.size();
    }

private:
<<<<<<< HEAD
    EOT _mean;
    EOT _variance;
=======
    ublas::vector< AtomType > _mean;
    ublas::symmetric_matrix< AtomType, ublas::lower > _varcovar;
>>>>>>> 36ec42d36204631eb4c25ae7b31a8728903697f8
};

#endif // !_doNormalParams_h
