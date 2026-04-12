/*
 <moRealVectorNeighbor.h>
 Copyright (C) Eremey Valetov, 2026

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

 As a counterpart to the access to the source code and  rights to copy,
 modify and redistribute granted by the license, users are provided only
 with a limited warranty  and the software's author,  the holder of the
 economic rights,  and the successive licensors  have only  limited liability.

 In this respect, the user's attention is drawn to the risks associated
 with loading,  using,  modifying and/or developing or reproducing the
 software by the user in light of its specific status of free software,
 that may mean  that it is complicated to manipulate,  and  that  also
 therefore means  that it is reserved for developers  and  experienced
 professionals having in-depth computer knowledge. Users are therefore
 encouraged to load and test the software's suitability as regards their
 requirements in conditions enabling the security of their systems and/or
 data to be ensured and,  more generally, to use and operate it in the
 same conditions as regards security.
 The fact that you are presently reading this means that you have had
 knowledge of the CeCILL license and that you accept its terms.

 ParadisEO WebSite : https://nojhan.github.io/paradiseo/
 Contact: https://github.com/nojhan/paradiseo/issues
*/

#ifndef _moRealVectorNeighbor_h
#define _moRealVectorNeighbor_h

#include <EO.h>
#include <neighborhood/moBackableNeighbor.h>
#include <vector>
#include <iterator>
#include <ostream>

/**
 * Neighbor for real-valued vector solutions.
 *
 * Stores a delta vector; move() adds it to the solution, moveBack() subtracts it.
 */
template<class EOT, class Fitness = typename EOT::Fitness>
class moRealVectorNeighbor : public moBackableNeighbor<EOT, Fitness>, public std::vector<double> {
public:
    typedef std::vector<double> EOVT;
    using EOVT::operator[];
    using EOVT::size;

    moRealVectorNeighbor(unsigned _size = 0, double _value = 0.0)
        : moBackableNeighbor<EOT, Fitness>(), EOVT(_size, _value) {}

    moRealVectorNeighbor(const moRealVectorNeighbor& _n)
        : moBackableNeighbor<EOT, Fitness>(_n), EOVT(_n) {}

    moRealVectorNeighbor& operator=(const moRealVectorNeighbor& _source) {
        moBackableNeighbor<EOT, Fitness>::operator=(_source);
        EOVT::operator=(_source);
        return *this;
    }

    bool equals(moRealVectorNeighbor& _neighbor) const {
        if (size() != _neighbor.size())
            return false;
        for (size_t i = 0; i < size(); i++)
            if ((*this)[i] != _neighbor[i])
                return false;
        return true;
    }

    void move(EOT& _solution) override {
        for (size_t i = 0; i < _solution.size(); i++)
            _solution[i] += (*this)[i];
        _solution.invalidate();
    }

    void moveBack(EOT& _solution) override {
        for (size_t i = 0; i < _solution.size(); i++)
            _solution[i] -= (*this)[i];
        _solution.invalidate();
    }

    void printOn(std::ostream& _os) const override {
        EO<Fitness>::printOn(_os);
        _os << ' ' << size() << ' ';
        std::copy(this->begin(), this->end(), std::ostream_iterator<double>(_os, " "));
    }

    std::string className() const override {
        return "moRealVectorNeighbor";
    }
};

#endif
