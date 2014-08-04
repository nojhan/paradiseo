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

#include "../eoBreed.h"
#include "../eoGeneralBreeder.h"
#include "../eoOneToOneBreeder.h"

#include "PyEO.h"
#include "def_abstract_functor.h"

using namespace boost::python;

#define DEF3(x, i1, i2)				\
    class_<x<PyEO>, bases<eoBreed<PyEO > > >	\
    (#x,					\
     init<i1, i2 >()				\
     [						\
      with_custodian_and_ward<1,2,		\
      with_custodian_and_ward<1,3		\
      >						\
      >						\
     ()						\
     ]						\
     )						\
    .def("__call__", &eoBreed<PyEO>::operator())

void breeders()
{
    def_abstract_functor<eoBreed<PyEO> >("eoBreed");

    DEF3(eoSelectTransform, eoSelect<PyEO>&, eoTransform<PyEO>&);

    DEF3(eoGeneralBreeder, eoSelectOne<PyEO>&, eoGenOp<PyEO>&)
        .def( init<eoSelectOne<PyEO>&, eoGenOp<PyEO>&, double>()[WC2])
        .def( init<eoSelectOne<PyEO>&, eoGenOp<PyEO>&, double, bool>()[WC2] )
        .def( init<eoSelectOne<PyEO>&, eoGenOp<PyEO>&, eoHowMany>() );


    DEF3(eoOneToOneBreeder, eoGenOp<PyEO>&, eoEvalFunc<PyEO>&)
        .def( init<eoGenOp<PyEO>&, eoEvalFunc<PyEO>&, double>()[WC2] )
        .def( init<eoGenOp<PyEO>&, eoEvalFunc<PyEO>&, double, eoHowMany>()[WC2] );
}
