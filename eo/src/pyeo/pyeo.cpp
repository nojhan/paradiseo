/*
    pyeo
    
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


#include <eoPop.h>
#include "pyeo.h"

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

	return make_tuple(object(_pop.size()), entries);
    }

    static void setstate( eoPop<PyEO>& _pop, boost::python::tuple pickled)
    {
	int sz = extract<int>(pickled[0]);
	boost::python::list entries = pickled[1];
	_pop.resize(sz);
	for (unsigned i = 0; i != _pop.size(); ++i)
	    PyEO_pickle_suite::setstate(_pop[i], tuple(entries[i]) );
    }
};


template <class T>
boost::python::str to_string(T& _p)
{
    std::ostrstream os;
    _p.printOn(os);
    os << ends;
    std::string s(os.str());
    return str(s.c_str());
}

void pop_sort(eoPop<PyEO>& pop) { pop.sort(); }
void pop_shuffle(eoPop<PyEO>& pop) { pop.shuffle(); }

struct index_error { index_error(std::string w) : what(w) {}; std::string what; };
void translate_index_error(index_error const& e)
{
        PyErr_SetString(PyExc_IndexError, e.what.c_str());
}

PyEO& pop_getitem(eoPop<PyEO>& pop, object key)
{
    extract<int> x(key);
    if (!x.check())
	throw index_error("Slicing not allowed");
    
    int i = x();
    
    if (static_cast<unsigned>(i) >= pop.size())
    { 
	throw index_error("Index out of bounds");
    }
    
    return pop[i];
}
void pop_setitem(eoPop<PyEO>& pop, object key, PyEO& value)
{

    extract<int> x(key);
    if (!x.check())
	throw index_error("Slicing not allowed");
    
    int i = x();
    
    if (static_cast<unsigned>(i) >= pop.size())
    { 
	throw index_error("Index out of bounds");
    }
    
    pop[i] = value;
}

void pop_push_back(eoPop<PyEO>& pop, PyEO& p) { pop.push_back(p); }
void pop_resize(   eoPop<PyEO>& pop, unsigned i) { pop.resize(i); }

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

BOOST_PYTHON_MODULE(pyeo)
{
    register_exception_translator<index_error>(&translate_index_error);
    
    class_<PyEO>("EO")
	.add_property("fitness", &PyEO::getFitness, &PyEO::setFitness)
	.add_property("genome", &PyEO::getGenome, &PyEO::setGenome)
	.def_pickle(PyEO_pickle_suite())
	.def("invalidate", &PyEO::invalidate)
	.def("invalid", &PyEO::invalid)
	.def("__str__", &PyEO::to_string)
	;

    class_<eoPop<PyEO> >("Pop", init<>() )
	.def( init< unsigned, eoInit<PyEO>& >() )
	.def("append", &eoPop<PyEO>::append)
	.def("__str__", to_string<eoPop<PyEO> >)
	.def("__len__", &eoPop<PyEO>::size)
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
    continuators();
    reduce();
    replacement();
    breeders();
    mergers();
    algos();

}


// to avoid having to build with libeo.a
ostream & operator << ( ostream& _os, const eoPrintable& _o ) {
            _o.printOn(_os);
	            return _os;
};
