// (c) Thales group, 2010
/*
    Authors:
	     Johann Dreo <johann.dreo@thalesgroup.com>
	     Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _edoNormalMulti_h
#define _edoNormalMulti_h

#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "edoDistrib.h"

namespace ublas = boost::numeric::ublas;

//! edoNormalMulti< EOT >

template < typename EOT >
class edoNormalMulti : public edoDistrib< EOT >
{
public:
    typedef typename EOT::AtomType AtomType;

    edoNormalMulti
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

    unsigned int size()
    {
	assert(_mean.size() == _varcovar.size1());
	assert(_mean.size() == _varcovar.size2());
	return _mean.size();
    }

    ublas::vector< AtomType > mean() const {return _mean;}
    ublas::symmetric_matrix< AtomType, ublas::lower > varcovar() const {return _varcovar;}

private:
    ublas::vector< AtomType > _mean;
    ublas::symmetric_matrix< AtomType, ublas::lower > _varcovar;
};

#endif // !_edoNormalMulti_h
