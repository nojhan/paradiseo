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

#include <eoSGA.h>
#include <eoEasyEA.h>
#include <eoDetSelect.h>
#include <eoCellularEasyEA.h>

#include "PyEO.h"
#include "def_abstract_functor.h"

using namespace boost::python;

void algos()
{
    def_abstract_functor<eoAlgo<PyEO> >("eoAlgo");

    /* Algorithms */
    class_<eoSGA<PyEO>, bases<eoAlgo<PyEO> >, boost::noncopyable>
        ("eoSGA",
         init<
         eoSelectOne<PyEO>&,
         eoQuadOp<PyEO>&,
         float,
         eoMonOp<PyEO>&,
         float,
         eoEvalFunc<PyEO>&,
         eoContinue<PyEO>&
         >()
         [
          with_custodian_and_ward<1,2,
          with_custodian_and_ward<1,3,
          with_custodian_and_ward<1,5,
          with_custodian_and_ward<1,7,
          with_custodian_and_ward<1,8>
          >
         >
          >
         >()
          ]
         )
        .def("__call__", &eoSGA<PyEO>::operator())
        ;

    class_<eoEasyEA<PyEO>, bases<eoAlgo<PyEO> > >
        ("eoEasyEA",
         init<
         eoContinue<PyEO>&,
         eoEvalFunc<PyEO>&,
         eoBreed<PyEO>&,
         eoReplacement<PyEO>&
         >()
         )
        .def( init<
              eoContinue<PyEO>&,
              eoEvalFunc<PyEO>&,
              eoBreed<PyEO>&,
              eoReplacement<PyEO>&,
              unsigned
              >() )
        .def( init<
              eoContinue<PyEO>&,
              eoPopEvalFunc<PyEO>&,
              eoBreed<PyEO>&,
              eoReplacement<PyEO>&
              >() )
        .def( init<
              eoContinue<PyEO>&,
              eoEvalFunc<PyEO>&,
              eoBreed<PyEO>&,
              eoMerge<PyEO>&,
              eoReduce<PyEO>&
              >() )
        .def( init<
              eoContinue<PyEO>&,
              eoEvalFunc<PyEO>&,
              eoSelect<PyEO>&,
              eoTransform<PyEO>&,
              eoReplacement<PyEO>&
              >() )
        .def( init<
              eoContinue<PyEO>&,
              eoEvalFunc<PyEO>&,
              eoSelect<PyEO>&,
              eoTransform<PyEO>&,
              eoMerge<PyEO>&,
              eoReduce<PyEO>&
              >() )
        .def("__call__", &eoEasyEA<PyEO>::operator())
        ;

    /*
      class_<eoCellularEasyEA<PyEO>, bases< eoAlgo<PyEO> > >("eoCellularEasyEA",
      init<   eoContinue<PyEO>&,
      eoEvalFunc<PyEO>&,
      eoSelectOne<PyEO>&,
      eoBinOp<PyEO>&,
      eoMonOp<PyEO>&,
      eoSelectOne<PyEO>&>())
      .def(
      init<   eoContinue<PyEO>&,
      eoEvalFunc<PyEO>&,
      eoSelectOne<PyEO>&,
      eoQuadOp<PyEO>&,
      eoMonOp<PyEO>&,
      eoSelectOne<PyEO>&>())
      ;
    */
}
