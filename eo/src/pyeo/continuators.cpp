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

#include <eoGenContinue.h>
#include <eoCombinedContinue.h>
#include <eoEvalContinue.h>
#include <eoFitContinue.h>
#include <eoSteadyFitContinue.h>

#include "pyeo.h"
#include "def_abstract_functor.h"

#define DEF(x) class_<x<PyEO>, bases<eoContinue<PyEO > > >(#x).def("__call__", &eoContinue<PyEO>::operator())
#define DEF2(x, i1) class_<x<PyEO>, bases<eoContinue<PyEO > > >(#x, init<i1>() ).def("__call__", &eoContinue<PyEO>::operator())
#define DEF3(x, i1, i2) class_<x<PyEO>, bases<eoContinue<PyEO > > >(#x, init<i1, i2 >() ).def("__call__", &eoContinue<PyEO>::operator())

void continuators()
{
    /* Counters, wrappers etc */
    
    class_<eoEvalFuncCounter<PyEO>, bases<eoEvalFunc<PyEO> > >("eoEvalFuncCounter",
	    init< eoEvalFunc<PyEO>&, std::string>())
	.def("__call__", &eoEvalFuncCounter<PyEO>::operator())
	;
    /* Continuators */
    def_abstract_functor<eoContinue<PyEO> >("eoContinue"); 
	
    class_<eoGenContinue<PyEO>, bases<eoContinue<PyEO> >, boost::noncopyable >("eoGenContinue", init<unsigned long>() )
	.def("__call__", &eoGenContinue<PyEO>::operator())
	;
  
    class_<eoCombinedContinue<PyEO>, bases<eoContinue<PyEO> > >("eoCombinedContinue", init<eoContinue<PyEO>&>())
	.def( init<eoContinue<PyEO>&, eoContinue<PyEO>& >() )
	.def("add", &eoCombinedContinue<PyEO>::add)
	.def("__call__", &eoCombinedContinue<PyEO>::operator())
	;
   
    class_<eoEvalContinue<PyEO>, bases<eoContinue<PyEO> > >("eoEvalContinue", 
	    init<eoEvalFuncCounter<PyEO>&, unsigned long>())
	.def("__call__", &eoEvalContinue<PyEO>::operator())
	;

    DEF2(eoFitContinue, object); // object is the fitness type

    DEF3(eoSteadyFitContinue, unsigned long, unsigned long);
}
