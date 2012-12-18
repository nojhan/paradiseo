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

#include <eoEvalFunc.h>
#include <eoInit.h>
#include <eoTransform.h>
#include <eoSGATransform.h>
#include <eoPopEvalFunc.h>

#include "PyEO.h"
#include "def_abstract_functor.h"

using namespace boost::python;

void abstract1()
{
    /* Abstract Classes: overrideble from python */
    def_abstract_functor<eoEvalFunc<PyEO> >("eoEvalFunc");
    def_abstract_functor<eoInit< PyEO > >("eoInit");

    def_abstract_functor<eoTransform<PyEO> >("eoTransform");

    class_<eoSGATransform<PyEO>, bases<eoTransform<PyEO> > >
        ("eoSGATransform",
         init<
         eoQuadOp<PyEO>&,
         double,
         eoMonOp<PyEO>&,
         double
         >()
         )
        .def("__call__", &eoSGATransform<PyEO>::operator());

    def_abstract_functor<eoPopEvalFunc<PyEO> >("eoPopEvalFunc");
}
