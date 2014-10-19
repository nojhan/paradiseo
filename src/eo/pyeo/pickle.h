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

#ifndef PICKLE_H
#define PICKLE_h

// #ifndef WIN32
// #include <config.h>
// #endif

#include <boost/python.hpp>
#include <sstream>
/** Implements pickle support for eoPersistent derivatives */

template <class T>
struct T_pickle_suite : boost::python::pickle_suite
{
    static
    std::string print_to_string(const T& t)
    {
        std::ostringstream os;
        t.printOn(os);
        os << std::ends;
        return os.str();
    }

    static
    boost::python::tuple getstate(const T& t)
    {
        std::string s = print_to_string(t);
        return boost::python::make_tuple( boost::python::str(s));
    }

    static
    void setstate(T& t, boost::python::tuple pickled)
    {
        std::string s = boost::python::extract<std::string>(pickled[0]);
        std::istringstream is(s);
        t.readFrom(is);
    }
};

/** Defines persistency through pickle support by using std::strings
 * so while we're at it, we will .def("__str__") as well
 */
template <class Persistent, class X1, class X2, class X3>
boost::python::class_<Persistent, X1, X2, X3>& pickle(boost::python::class_<Persistent, X1, X2, X3>& c)
{
    return c.def_pickle(T_pickle_suite<Persistent>())
        .def("__str__", T_pickle_suite<Persistent>::print_to_string);
}

#endif
