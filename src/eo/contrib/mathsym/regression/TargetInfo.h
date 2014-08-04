/*	    
 *             Copyright (C) 2005 Maarten Keijzer
 *
 *          This program is free software; you can redistribute it and/or modify
 *          it under the terms of version 2 of the GNU General Public License as 
 *          published by the Free Software Foundation. 
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program; if not, write to the Free Software
 *          Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef TARGETINFO_H_
#define TARGETINFO_H_

#include <valarray>
#include <vector>

class TargetInfo {
    std::valarray<double> _targets;
    std::valarray<double> _weights;
    std::valarray<bool>   _training_mask;

    // some stuff for ols
    std::valarray<double> _tcov_part;
    double _tmean;
    double _tvar;
    double _tstd;
    double _tmed;

    public:
    TargetInfo() {}

    TargetInfo(const std::valarray<double>& t);
    TargetInfo(const std::valarray<double>& t, const std::valarray<double>& w);

    TargetInfo(const TargetInfo& org);
    TargetInfo& operator=(const TargetInfo& org);
    ~TargetInfo() {}

    const std::valarray<double>& targets() const { return _targets; }
    const std::valarray<double>& weights() const { return _weights; }
    const std::valarray<bool>&   mask() const { return _training_mask; }

    void set_training_mask(const std::valarray<bool>& mask);

    bool has_weights() const { return _weights.size(); }
    bool has_mask()    const { return _training_mask.size(); }

    std::vector<int> sort();

    const std::valarray<double>& tcov_part() const { return _tcov_part; }
    double tmean() const { return _tmean; }
    double tvar()  const { return _tvar; }
    double tstd()  const { return _tstd; }
    double devmedian() const { return _tmed; }
};

#endif

