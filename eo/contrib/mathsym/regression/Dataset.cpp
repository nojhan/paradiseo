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

#include "Dataset.h"
#include <fstream>
#include <sstream>

#include <iostream>

using namespace std;

class DataSetImpl {
    public: 
    vector< vector<double> > inputs;
    vector<double> targets;

    void read_data(vector<string> strings) {
	// find the number of inputs
	
	istringstream cnt(strings[0]);
	unsigned n = 0;
	for (;;) {
	    string s;
	    cnt >> s;
	    if (!cnt) break;
	    ++n;
	}

	inputs.resize(strings.size(), vector<double>(n-1));
	targets.resize(strings.size());

	for (unsigned i = 0; i < strings.size(); ++i) {
	    istringstream is(strings[i]);
	    for (unsigned j = 0; j < n; ++j) {
		
		if (!is) {
		    cerr << "Too few targets in record " << i << endl;
		    exit(1);
		}
		
		if (j < n-1) {
		    is >> inputs[i][j];
		} else {
		    is >> targets[i];
		}
		
	    }
	}
	
    }
    
};

Dataset::Dataset() { pimpl = new DataSetImpl; }
Dataset::~Dataset() { delete pimpl; }
Dataset::Dataset(const Dataset& that) { pimpl = new DataSetImpl(*that.pimpl); }
Dataset& Dataset::operator=(const Dataset& that) { *pimpl = *that.pimpl; return *this; }

unsigned Dataset::n_records() const { return pimpl->targets.size(); }
unsigned Dataset::n_fields()  const { return pimpl->inputs[0].size(); }
const std::vector<double>& Dataset::get_inputs(unsigned record) const { return pimpl->inputs[record]; }
double Dataset::get_target(unsigned record) const { return pimpl->targets[record]; }

double error(string errstr);

void Dataset::load_data(std::string filename) {
    vector<string> strings; // first load it in strings

    ifstream is(filename.c_str());

    for(;;) {
	string s;
	getline(is, s);
	if (!is) break;

	if (s[0] == '#') continue; // comment, skip

	strings.push_back(s);
    }
   
    is.close();

    if (strings.size() == 0) {
	error("No data could be loaded");
    }
    
    pimpl->read_data(strings);
    
}

std::vector<double> Dataset::input_minima() const {
    vector<vector<double> >& in = pimpl->inputs;
    
    vector<double> mn(in[0].size(), 1e+50);
    for (unsigned i = 0; i < in.size(); ++i) {
	for (unsigned j = 0; j < in[i].size(); ++j) {
	    mn[j] = std::min(mn[j], in[i][j]);
	}
    }
    
    return mn;
}

vector<double> Dataset::input_maxima() const {
    vector<vector<double> >& in = pimpl->inputs;
    
    vector<double> mx(in[0].size(), -1e+50);
    for (unsigned i = 0; i < in.size(); ++i) {
	for (unsigned j = 0; j < in[i].size(); ++j) {
	    mx[j] = std::max(mx[j], in[i][j]);
	}
    }
    
    return mx;
}




