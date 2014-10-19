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

#ifndef DATASET_H_
#define DATASET_H_

#include <string>
#include <vector>

class DataSetImpl;

class Dataset {
    
    DataSetImpl* pimpl;
    
    Dataset& operator=(const Dataset&); // cannot assign
    public:

    Dataset();
    ~Dataset();
    Dataset(const Dataset&);

    void load_data(std::string filename);
    
    unsigned n_records() const;
    unsigned n_fields() const;

    const std::vector<double>& get_inputs(unsigned record) const;
    double get_target(unsigned record) const;

    std::vector<double> input_minima() const;
    std::vector<double> input_maxima() const;
    
};

#endif

