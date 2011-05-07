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

#include <eoNDSorting.h>

#include "PyEO.h"

using namespace boost::python;

struct Perf2WorthWrapper : public eoPerf2Worth<PyEO,double>
{
    PyObject* self;
    Perf2WorthWrapper(PyObject* p) : self(p) {}

    void operator()( const eoPop<PyEO>& pop)
    {
        call_method<void>(self, "__call__", boost::ref(pop));
    }
};

numeric::array get_worths(eoPerf2Worth<PyEO, double>& p)
{
    std::vector<double>& worths = p.value();
    list result;

    for (unsigned i = 0; i < worths.size(); ++i)
        result.append(worths[i]);

    return numeric::array(result);
}

struct CachedPerf2WorthWrapper : public eoPerf2WorthCached<PyEO, double>
{
    PyObject* self;
    CachedPerf2WorthWrapper(PyObject* p) : self(p) {}

    void calculate_worths(const eoPop<PyEO>& pop)
    {
        call_method<void>(self, "calculate_worths", boost::ref(pop));
    }
};

void perf2worth()
{
    //numeric::array::set_module_and_type("Numeric", "ArrayType");

    class_<eoPerf2Worth<PyEO, double>,
        Perf2WorthWrapper,
        bases< eoValueParam<std::vector<double> > >,
        boost::noncopyable>("eoPerf2Worth", init<>())

        .def("__call__", &Perf2WorthWrapper::operator())
        .def("sort_pop", &eoPerf2Worth<PyEO, double>::sort_pop)
    //.def("value", get_worths)
        ;

    class_<eoPerf2WorthCached<PyEO, double>,
        CachedPerf2WorthWrapper,
        bases<eoPerf2Worth<PyEO, double> >,
        boost::noncopyable>("eoPerf2WorthCached", init<>())

        .def("__call__", &eoPerf2WorthCached<PyEO, double>::operator())
        .def("calculate_worths", &CachedPerf2WorthWrapper::calculate_worths)
        ;

    //class_<eoNoPerf2Worth<PyEO>, bases<eoPerf2Worth<PyEO, double> > >("eoNoPerf2Worth")
    //	.def("__call__", &eoNoPerf2Worth<PyEO>::operator());

    class_<eoNDSorting_II<PyEO>,
        bases<eoPerf2WorthCached<PyEO, double> > >("eoNDSorting_II")
        .def("calculate_worths", &eoNDSorting_II<PyEO>::calculate_worths);
}
