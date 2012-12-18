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

#include <eoSelect.h>
#include <eoDetSelect.h>
#include <eoSelectMany.h>
#include <eoSelectNumber.h>
#include <eoSelectPerc.h>
#include <eoTruncSelect.h>
#include <eoTruncatedSelectMany.h>

#include "PyEO.h"
#include "def_abstract_functor.h"

using namespace boost::python;

#define DEF(x) class_<x<PyEO>, bases<eoSelect<PyEO > > >(#x).def("__call__", &eoSelect<PyEO>::operator())
#define DEF2(x, i1) class_<x<PyEO>, bases<eoSelect<PyEO > > >(#x, init<i1>()[WC1] ).def("__call__", &eoSelect<PyEO>::operator())
#define DEF3(x, i1, i2) class_<x<PyEO>, bases<eoSelect<PyEO > > >(#x, init<i1, i2 >()[WC1] ).def("__call__", &eoSelect<PyEO>::operator())

void selectors()
{
    def_abstract_functor<eoSelect<PyEO> >("eoSelect");

    DEF(eoDetSelect).def( init<double>() ).def( init<double, bool>() );
    DEF3(eoSelectMany, eoSelectOne<PyEO>&, double)
	.def( init< eoSelectOne<PyEO>&, double, bool>()[WC1] )
	.def( init< eoSelectOne<PyEO>&, eoHowMany>()[WC1] );

    DEF2(eoSelectNumber, eoSelectOne<PyEO>&)
	.def( init< eoSelectOne<PyEO>&, unsigned>()[WC1]);

    DEF2(eoSelectPerc, eoSelectOne<PyEO>&)
	.def( init<eoSelectOne<PyEO>&, float>()[WC1] );

    DEF3(eoTruncSelect, eoSelectOne<PyEO>&, eoHowMany);

    class_<eoTruncatedSelectMany<PyEO>, bases<eoSelect<PyEO> > >("eoTruncatedSelectMany",
								 init<eoSelectOne<PyEO>&, double, double>()[WC1])
	.def(init<eoSelectOne<PyEO>&, double, double, bool> ()[WC1])
	.def(init<eoSelectOne<PyEO>&, double, double, bool, bool> ()[WC1])
	.def(init<eoSelectOne<PyEO>&, eoHowMany, eoHowMany> ()[WC1]);
}
