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

#include "../eoReduce.h"

#include "PyEO.h"

using namespace boost::python;

// unfortunately have to define it specially
class eoReduceWrapper : public eoReduce<PyEO>
{
public:
    PyObject* self;
    eoReduceWrapper(PyObject* s) : self(s) {}
    void operator()(eoPop<PyEO>& pop, unsigned i)
    {
	boost::python::call_method<void>(self, "__call__", pop, i );
    }
};

void reduce()
{
    // ref trick in def_abstract_functor does not work for unsigned int :-(
    class_<eoReduce<PyEO>, eoReduceWrapper, boost::noncopyable>("eoReduce", init<>())
	.def("__call__", &eoReduceWrapper::operator());

    class_<eoTruncate<PyEO>, bases<eoReduce<PyEO> > >("eoTruncate", init<>() )
	.def("__call__", &eoReduce<PyEO>::operator())
	;
    class_<eoRandomReduce<PyEO>, bases<eoReduce<PyEO> > >("eoRandomReduce")
	.def("__call__", &eoReduce<PyEO>::operator())
	;
    class_<eoEPReduce<PyEO>, bases<eoReduce<PyEO> > >("eoEPReduce", init<unsigned>())
	.def("__call__", &eoReduce<PyEO>::operator())
	;
    class_<eoLinearTruncate<PyEO>, bases<eoReduce<PyEO> > >("eoLinearTruncate")
	.def("__call__", &eoReduce<PyEO>::operator())
	;
    class_<eoDetTournamentTruncate<PyEO>, bases<eoReduce<PyEO> > >("eoDetTournamentTruncate", init<unsigned>())
	.def("__call__", &eoReduce<PyEO>::operator())
	;
    class_<eoStochTournamentTruncate<PyEO>, bases<eoReduce<PyEO> > >("eoStochTournamentTruncate", init<double>())
	.def("__call__", &eoReduce<PyEO>::operator())
	;
}
