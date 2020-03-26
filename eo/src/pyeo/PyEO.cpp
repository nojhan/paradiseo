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

#include <sstream>

#include "PyEO.h"
#include "../eoPop.h"

using namespace std;
//using namespace boost::python;

// static member, needs to be instantiated somewhere
std::vector<int> PyFitness::objective_info;

bool PyFitness::dominates(const PyFitness& oth) const
{
    bool dom = false;

    for (unsigned i = 0; i < nObjectives(); ++i)
        {
            int objective = objective_info[i];

            if (objective == 0) // ignore
                continue;

            bool maxim = objective > 0;

            double aval = maxim? (*this)[i] : -(*this)[i];
            double bval = maxim? oth[i] : -oth[i];

            if (fabs(aval - bval) > tol())
                {
                    if (aval < bval)
                        {
                            return false; // cannot dominate
                        }
                    // else aval < bval
                    dom = true; // for the moment: goto next objective
                }
            //else they're equal in this objective, goto next
        }

    return dom;
}

bool dominates(const PyEO& a, const PyEO& b)
{
    return PyFitness(a.fitness()).dominates(b.fitness());
}

ostream& operator<<(ostream& os, const PyEO& _eo)
{
    os << _eo.to_string();
    return os;
}

struct pyPop_pickle_suite : boost::python::pickle_suite
{
    static boost::python::tuple getstate(const eoPop<PyEO>& _pop)
    {
        boost::python::list entries;
        for (unsigned i = 0; i != _pop.size(); ++i)
            entries.append( PyEO_pickle_suite::getstate(_pop[i]) );

        return boost::python::make_tuple(boost::python::object(_pop.size()), entries);
    }

    static void setstate( eoPop<PyEO>& _pop, boost::python::tuple pickled)
    {
        int sz = boost::python::extract<int>(pickled[0]);
        boost::python::list entries = boost::python::list(pickled[1]);
        _pop.resize(sz);
        for (unsigned i = 0; i != _pop.size(); ++i)
            {
                PyEO_pickle_suite::setstate(_pop[i], boost::python::tuple(entries[i]) );
            }
    }
};


template <class T>
boost::python::str to_string(T& _p)
{
    std::ostringstream os;
    _p.printOn(os);
    return boost::python::str(os.str().c_str());
}

void pop_sort(eoPop<PyEO>& pop) { pop.sort(); }
void pop_shuffle(eoPop<PyEO>& pop) { pop.shuffle(); }

void translate_index_error(index_error const& e)
{
    PyErr_SetString(PyExc_IndexError, e.what.c_str());
}

PyEO& pop_getitem(eoPop<PyEO>& pop, boost::python::object key)
{
    boost::python::extract<int> x(key);
    if (!x.check())
        throw eoException("Slicing not allowed");

    int i = x();

    if (static_cast<unsigned>(i) >= pop.size())
        {
            throw eoException("Index out of bounds");
        }
    return pop[i];
}

void pop_setitem(eoPop<PyEO>& pop, boost::python::object key, PyEO& value)
{
    boost::python::extract<int> x(key);
    if (!x.check())
        throw eoException("Slicing not allowed");

    int i = x();

    if (static_cast<unsigned>(i) >= pop.size())
        {
            throw eoException("Index out of bounds");
        }

    pop[i] = value;
}

void pop_push_back(eoPop<PyEO>& pop, PyEO& p) { pop.push_back(p); }
void pop_resize(   eoPop<PyEO>& pop, unsigned i) { pop.resize(i); }
int pop_size(   eoPop<PyEO>& pop) { return pop.size(); }

extern void abstract1();
extern void algos();
extern void random_numbers();
extern void geneticOps();
extern void selectOne();
extern void continuators();
extern void reduce();
extern void replacement();
extern void selectors();
extern void breeders();
extern void mergers();
extern void valueParam();
extern void perf2worth();
extern void monitors();
extern void statistics();

BOOST_PYTHON_MODULE(libPyEO)
{
    using namespace boost::python;

    boost::python::register_exception_translator<index_error>(&translate_index_error);

    boost::python::class_<PyEO>("EO")
        .add_property("fitness", &PyEO::getFitness, &PyEO::setFitness)
        .add_property("genome", &PyEO::getGenome, &PyEO::setGenome)
        .def_pickle(PyEO_pickle_suite())
        .def("invalidate", &PyEO::invalidate)
        .def("invalid", &PyEO::invalid)
        .def("__str__", &PyEO::to_string)
        ;

    boost::python::class_<eoPop<PyEO> >("eoPop", init<>() )
        .def( init< unsigned, eoInit<PyEO>& >()[with_custodian_and_ward<1,3>()] )
        .def("append", &eoPop<PyEO>::append, "docstring?")
        .def("__str__", to_string<eoPop<PyEO> >)
        .def("__len__", pop_size)
        .def("sort",    pop_sort )
        .def("shuffle", pop_shuffle)
        .def("__getitem__", pop_getitem, return_internal_reference<>() )
        .def("__setitem__", pop_setitem)
        .def("best", &eoPop<PyEO>::best_element, return_internal_reference<>() )
        .def("push_back", pop_push_back)
        .def("resize",    pop_resize)
        .def_pickle(pyPop_pickle_suite())
        ;


    // Other definitions in different compilation units,
    // this to avoid having g++ to choke on the load
    random_numbers();
    valueParam();
    abstract1();
    geneticOps();
    selectOne();
    selectors();
    perf2worth();
    monitors();
    statistics();
    continuators();
    reduce();
    replacement();
    breeders();
    mergers();
    algos();

    // The traits class
    class_<PyFitness>("PyFitness");

    def("nObjectives", &PyFitness::nObjectives);
    def("tol", &PyFitness::tol);
    def("maximizing", &PyFitness::maximizing);
    def("setObjectivesSize", &PyFitness::setObjectivesSize);
    def("setObjectivesValue", &PyFitness::setObjectivesValue);
    def("dominates", dominates);
}


// to avoid having to build with libeo.a
ostream & operator << ( ostream& _os, const eoPrintable& _o )
{
    _o.printOn(_os);
    return _os;
};
