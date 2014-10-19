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

#include "../eoSelectOne.h"
#include "../eoDetTournamentSelect.h"
#include "../eoRandomSelect.h"
#include "../eoStochTournamentSelect.h"
#include "../eoTruncatedSelectOne.h"
#include "../eoSequentialSelect.h"

#include "PyEO.h"
#include "pickle.h"
#include "def_abstract_functor.h"

using namespace boost::python;

class eoSelectOneWrapper : public eoSelectOne<PyEO>
{
public:
    PyObject* self;
    eoSelectOneWrapper(PyObject* p) : self(p) {}
    const PyEO& operator()(const eoPop<PyEO>& pop)
    {
	return boost::python::call_method< const PyEO& >(self, "__call__", boost::ref(pop));
    }
};

template <class Select>
void add_select(std::string name)
{
    class_<Select, bases<eoSelectOne<PyEO> > >(name.c_str(), init<>() )
	.def("__call__", &Select::operator(), return_internal_reference<>() )
	;
}

template <class Select, class Init>
void add_select(std::string name, Init init)
{
    class_<Select, bases<eoSelectOne<PyEO> > >(name.c_str(), init)
	.def("__call__", &Select::operator(), return_internal_reference<>() )
	;
}

template <class Select, class Init1, class Init2>
void add_select(std::string name, Init1 init1, Init2 init2)
{
    class_<Select, bases<eoSelectOne<PyEO> > >(name.c_str(), init1)
	.def( init2 )
	.def("__call__", &Select::operator(), return_internal_reference<>() )
	.def("setup", &Select::setup);
}

void selectOne()
{
    /* Concrete classes */

    pickle(class_<eoHowMany>("eoHowMany", init<>())
	   .def( init<double>() )
	   .def( init<double, bool>() )
	   .def( init<int>() )
	   .def("__call__", &eoHowMany::operator())
	   .def("__neg__", &eoHowMany::operator-)
	   );

    class_<eoSelectOne<PyEO>, eoSelectOneWrapper, boost::noncopyable>("eoSelectOne", init<>())
	.def("__call__", &eoSelectOneWrapper::operator(), return_internal_reference<>() )
	.def("setup", &eoSelectOne<PyEO>::setup);

    /* SelectOne derived classes */

    add_select<eoDetTournamentSelect<PyEO> >("eoDetTournamentSelect", init<>(), init<unsigned>() );
    add_select<eoStochTournamentSelect<PyEO> >("eoStochTournamentSelect", init<>(), init<double>() );
    add_select<eoTruncatedSelectOne<PyEO> >("eoTruncatedSelectOne",
					    init<eoSelectOne<PyEO>&, double>()[WC1],
					    init<eoSelectOne<PyEO>&, eoHowMany >()[WC1]
					    );

    // eoProportionalSelect is not feasible to implement at this point as fitness is not recognizable as a float
    // use eoDetTournament instead: with a t-size of 2 it is equivalent to eoProportional with linear scaling
    //add_select<eoProportionalSelect<PyEO> >("eoProportionalSelect", init<eoPop<PyEO>&>() );

    add_select<eoRandomSelect<PyEO> >("eoRandomSelect");
    add_select<eoBestSelect<PyEO> >("eoBestSelect");
    add_select<eoNoSelect<PyEO> >("eoNoSelect");

    add_select<eoSequentialSelect<PyEO> >("eoSequentialSelect", init<>(), init<bool>());
    add_select<eoEliteSequentialSelect<PyEO> >("eoEliteSequentialSelect");
    /*
     * eoSelectFromWorth.h:class eoSelectFromWorth : public eoSelectOne<EOT>
     */
}
