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

#include <eoMerge.h>
#include "PyEO.h"
#include "def_abstract_functor.h"

using namespace boost::python;

#define DEF(x) class_<x<PyEO>, bases<eoMerge<PyEO > > >(#x).def("__call__", &eoMerge<PyEO>::operator())
#define DEF2(x, i1) class_<x<PyEO>, bases<eoMerge<PyEO > > >(#x, init<i1>() ).def("__call__", &eoMerge<PyEO>::operator())
#define DEF3(x, i1, i2) class_<x<PyEO>, bases<eoMerge<PyEO > > >(#x, init<i1, i2 >() ).def("__call__", &eoMerge<PyEO>::operator())

void mergers()
{
    def_abstract_functor<eoMerge<PyEO> >("eoMerge");

    DEF2(eoElitism, double)
        .def( init<double, bool>() );
    DEF(eoNoElitism);
    DEF(eoPlus);

}
