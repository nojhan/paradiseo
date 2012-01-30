/*
    PyEO

    Copyright (C) 2003 Maarten Keijzer

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#ifndef PYEO_H
#define PYEO_H

#include <string>
#include <vector>
#include <exception>
#include <boost/python.hpp>

#include <EO.h>

struct index_error : public std::exception
{
    index_error(std::string w) : what(w) {};
    virtual ~index_error() throw() {}
    std::string what;
};

class PyFitness : public boost::python::object
{
public :

    typedef PyFitness fitness_traits; // it's its own traits class :-)

    PyFitness() : boost::python::object() {}

    template <class T>
    PyFitness(const T& o) : boost::python::object(o) {}

    static unsigned nObjectives() { return objective_info.size(); }
    static double tol() { return 1e-6; }
    static bool maximizing(int which) { return objective_info[which] > 0; }

    static void setObjectivesSize(int sz) { objective_info.resize(sz, 0); }
    static void setObjectivesValue(unsigned which, int value)
    {
        if (which >= objective_info.size())
            {
                throw index_error("Too few elements allocated, resize objectives first");
            }

        objective_info[which] = value;
    }

    static std::vector<int> objective_info;

    bool dominates(const PyFitness& oth) const;

    double operator[](int i) const
    {
        boost::python::extract<double> x(object::operator[](i));

        if (!x.check())
            throw std::runtime_error("PyFitness: does not contain doubles");
        return x();
    }

    bool operator<(const PyFitness& other) const
    {
        if (objective_info.size() == 0)
            {
                const object& self = *this;
                const object& oth = other;
                return self < oth;
            }
        // otherwise use objective_info

        for (unsigned i = 0; i < objective_info.size(); ++i)
            {
                double a = objective_info[i] * operator[](i);
                double b = objective_info[i] * other[i];

                if ( fabs(a - b) > tol())
                    {
                        if (a < b)
                            return true;
                        return false;
                    }
            }

        return false;
    }

    bool operator>(const PyFitness& other) const
    {
        return other.operator<(*this);
    }

    void printOn(std::ostream& os) const { const boost::python::object& o = *this; boost::python::api::operator<<(os,o); }
    friend std::ostream& operator<<(std::ostream& os, const PyFitness& p) { p.printOn(os); return os;  }
    friend std::istream& operator>>(std::istream& is, PyFitness& p) { boost::python::object o; is >> o; p = o; return is; }
};

struct PyEO : public EO< PyFitness  >
{
    typedef PyFitness Fitness;

    boost::python::object getFitness() const { return invalid()? Fitness(): fitness(); }
    void setFitness(boost::python::object f) { if (f == Fitness()) invalidate(); else fitness(f); }

    boost::python::object getGenome() const { return genome; }
    void setGenome(boost::python::object g) { genome = g; }
    boost::python::object genome;

    std::string to_string() const
    {
        std::string result;
        result += boost::python::extract<const char*>(boost::python::str(getFitness()));
        result += ' ';
        result += boost::python::extract<const char*>(boost::python::str(genome));
        return result;
    }

    bool operator<(const PyEO& other) const { return EO<Fitness>::operator<(other); }
    bool operator>(const PyEO& other) const { return EO<Fitness>::operator>(other); }

};

std::ostream& operator<<(std::ostream& os, const PyEO& _eo);

struct PyEO_pickle_suite : boost::python::pickle_suite
{
    typedef PyEO::Fitness Fitness;

    static boost::python::tuple getstate(const PyEO& _eo)
    {
        return boost::python::make_tuple(_eo.getFitness(), _eo.genome);
    }

    static void setstate(PyEO& _eo, boost::python::tuple pickled)
    {
        _eo.setFitness( Fitness(pickled[0]) );
        _eo.genome = pickled[1];
    }
};

#endif
