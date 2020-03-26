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

#include "../utils/eoParam.h"
#include "../utils/eoMonitor.h"
#include "PyEO.h"

using namespace boost::python;

class MonitorWrapper : public eoMonitor
{
public:
    PyObject* self;
    list objects;

    MonitorWrapper(PyObject* p) :self(p) {}

    eoMonitor& operator()()
    {
        call_method<void>(self, "__call__");
        return *this;
    }

    std::string getString(int i)
    {
        if (static_cast<unsigned>(i) >= vec.size())
            {
                throw eoException("Index out of bounds");
            }

        return vec[i]->getValue();
    }

    unsigned size() { return vec.size(); }
};

void monitors()
{
    /**
     * Change of interface: I encountered some difficulties with
     * transferring eoParams from and to Python, so now we can
     * only get at the strings contained in the eoParams.
     * sorry
     */

    class_<eoMonitor, MonitorWrapper, boost::noncopyable>("eoMonitor", init<>())
        .def("lastCall", &eoMonitor::lastCall)
        .def("add", &eoMonitor::add)
        .def("__call__", &MonitorWrapper::operator(), return_internal_reference<1>() )
        .def("__getitem__", &MonitorWrapper::getString,
             "Returns the string value of the indexed Parameter")
        .def("__len__", &MonitorWrapper::size)
        ;
}
