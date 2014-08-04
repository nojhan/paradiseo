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

#include "TargetInfo.h"

using namespace std;

TargetInfo::TargetInfo(const TargetInfo& org) { operator=(org); }

TargetInfo& TargetInfo::operator=(const TargetInfo& org) {
    _targets.resize(org._targets.size());
    _weights.resize(org._weights.size());
    _tcov_part.resize(org._tcov_part.size());
    
    _targets = org._targets;
    _weights = org._weights;
    _tcov_part = org._tcov_part;

    _tmean = org._tmean;
    _tvar  = org._tvar;
    _tstd  = org._tstd;
    _tmed = org._tmed;
    return *this;
}
      

TargetInfo::TargetInfo(const std::valarray<double>& t) {
    _weights.resize(0);
    _targets.resize(t.size());
    _targets = t;
    
    _tmean = _targets.sum()/_targets.size();
    
    _tcov_part.resize(_targets.size());
    _tcov_part = _targets;
    _tcov_part -= _tmean;
	
    std::valarray<double> tmp = _tcov_part;
    tmp = _tcov_part;
    tmp *= tmp;
	
    _tvar = tmp.sum() / (tmp.size()-1);
    _tstd = sqrt(_tvar);
    _tmed = 0;
}

TargetInfo::TargetInfo(const std::valarray<double>& t, const std::valarray<double>& w) {

    _targets.resize(t.size());
    _weights.resize(w.size());

    _targets = t;
    _weights = w;
    
    double sumw = _weights.sum();
	// scale weights so that they'll add up to 1
    _weights /= sumw;
	
    _tmean = (_targets * _weights).sum();
    _tcov_part.resize(_targets.size());
    _tcov_part = _targets;
    _tcov_part -= _tmean;

    _tvar = (pow(_targets - _tmean, 2.0) * _weights).sum();
    _tstd = sqrt(_tvar);
    _tmed = 0.;
}

// calculate the members, now in the context of a mask
void TargetInfo::set_training_mask(const std::valarray<bool>& tmask) {
    
    TargetInfo tmp;
    
    if (has_weights() ) {
	tmp = TargetInfo( _targets[tmask], _weights[tmask]);
    } else {
	tmp = TargetInfo( _targets[tmask] );
    }
    
    _tcov_part.resize(tmp._tcov_part.size());
    _tcov_part = tmp._tcov_part;

    _tmean = tmp._tmean;
    _tvar  = tmp._tvar;
    _tstd  = tmp._tstd;
    _tmed =  tmp._tmed;

    _training_mask.resize(tmask.size());
    _training_mask = tmask;
}

struct SortOnTargets
{
    const valarray<double>& t;
    SortOnTargets(const valarray<double>& v) : t(v) {}

    bool operator()(int i, int j) const {
	return fabs(t[i]) < fabs(t[j]);
    }
};
    
vector<int> TargetInfo::sort() {
    
    vector<int> ind(_targets.size());
    for (unsigned i = 0; i < ind.size(); ++i) { ind[i] = i; }

    std::sort(ind.begin(), ind.end(), SortOnTargets(_targets));

    valarray<double> tmptargets = _targets;
    valarray<double> tmpweights = _weights;
    valarray<double> tmpcov     = _tcov_part;
    
    for (unsigned i = 0; i < ind.size(); ++i) 
    {
	_targets[i] = tmptargets[ ind[i] ];
	_tcov_part[i] = tmpcov[ ind[i] ];	
	if (_weights.size()) _weights[i] = tmpweights[ ind[i] ];
    }

    return ind;
}



