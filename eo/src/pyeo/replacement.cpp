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

#include "../eoReplacement.h"
#include "../eoMergeReduce.h"
#include "../eoReduceMerge.h"
#include "../eoReduceMergeReduce.h"
#include "../eoMGGReplacement.h"

#include "PyEO.h"
#include "def_abstract_functor.h"

using namespace boost::python;

#define DEF(x) class_<x<PyEO>, bases<eoReplacement<PyEO > > >(#x).def("__call__", &eoReplacement<PyEO>::operator())
#define DEF2(x, i1) class_<x<PyEO>, bases<eoReplacement<PyEO > > >(#x, init<i1>() ).def("__call__", &eoReplacement<PyEO>::operator())
#define DEF3(x, i1, i2) class_<x<PyEO>, bases<eoReplacement<PyEO > > >	\
    (#x,								\
     init<i1, i2 >() [WC2])						\
    .def("__call__", &eoReplacement<PyEO>::operator())

void replacement()
{
    def_abstract_functor<eoReplacement<PyEO> >("eoReplacement");

    // eoReplacement.h
    DEF(eoGenerationalReplacement);

    class_<eoWeakElitistReplacement<PyEO>, bases<eoReplacement<PyEO> > >
	("eoWeakElitistReplacement",
	 init< eoReplacement<PyEO>& >()[WC1]);

    // eoMergeReduce.h
    DEF3(eoMergeReduce, eoMerge<PyEO>&, eoReduce<PyEO>& );
    DEF(eoPlusReplacement);
    DEF(eoCommaReplacement);
    DEF2(eoEPReplacement, unsigned);

    // eoReduceMerge.h
    DEF3(eoReduceMerge, eoReduce<PyEO>&, eoMerge<PyEO>& );
    DEF(eoSSGAWorseReplacement);
    DEF2(eoSSGADetTournamentReplacement, unsigned);
    DEF2(eoSSGAStochTournamentReplacement, double);

    // eoReduceMergeReduce.h
    //class_<eoReduceMergeReduce<PyEO>, bases<eoReplacement<PyEO> > >("eoReplacement",
    //	    init<eoHowMany, bool, eoHowMany, eoReduce<PyEO>&,
    //		 eoHowMany, eoReduce<PyEO>&, eoReduce<PyEO>&>())
    //	.def("__call__", &eoReplacement<PyEO>::operator());

    //eoMGGReplacement
    DEF(eoMGGReplacement)
	.def( init<eoHowMany>() )
	.def( init<eoHowMany, unsigned>() );
}
