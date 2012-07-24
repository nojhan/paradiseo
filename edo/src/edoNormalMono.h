/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
    Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _edoNormalMono_h
#define _edoNormalMono_h

#include "edoDistrib.h"

/** @defgroup Mononormal Normal
 * A normal (Gaussian) distribution that only model variances of variables.
 *
 * @ingroup Distributions
 */

/** A normal (Gaussian) distribution that only model variances of variables.
 *
 * This is basically a mean vector and a variances vector. Do not model co-variances.
 *
 * @ingroup Distributions
 * @ingroup Mononormal
 */
template < typename EOT >
class edoNormalMono : public edoDistrib< EOT >
{
public:    
    edoNormalMono()
      : _mean(EOT(1,0)), _variance(EOT(1,1))
    {}

    edoNormalMono( const EOT& mean, const EOT& variance )
        : _mean(mean), _variance(variance)
    {
        assert(_mean.size() > 0);
        assert(_mean.size() == _variance.size());
    }

    unsigned int size()
    {
        assert(_mean.size() == _variance.size());
        return _mean.size();
    }

    EOT mean(){return _mean;}
    EOT variance(){return _variance;}

private:
    EOT _mean;
    EOT _variance;
};

#endif // !_edoNormalMono_h
