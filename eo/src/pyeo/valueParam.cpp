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

#include <utils/eoParam.h>
#include <boost/python.hpp>

using namespace boost::python;

class ParamWrapper : public eoParam
{
public:
    PyObject* self;
    ParamWrapper(PyObject* p) : self(p) {}
    ParamWrapper(PyObject* p,
	std::string a,
	std::string b,
	std::string c,
	char d,
	bool e) : eoParam(a,b,c,d,e), self(p) {}

    
    std::string getValue() const
    {
	return call_method<std::string>(self, "getValue");
    }

    void setValue(std::string s)
    {
	call_method<void>(self, "setValue", s);
    }
	
};

template <class T> T getv(const eoValueParam<T>& v)    { return v.value(); }
template <class T> void setv(eoValueParam<T>& v, T val) { v.value() = val; }

template <class T>
void define_valueParam(std::string prefix)
{
    class_<eoValueParam<T>, bases<eoParam> >( (prefix + "ValueParam").c_str(), init<>())
	.def(init<T, std::string, std::string, char, bool>())
	.def(init<T, std::string, std::string, char>())
	.def(init<T, std::string, std::string>())
	.def(init<T, std::string>())
	.def("getValue", &eoValueParam<T>::getValue)
	.def("__str__", &eoValueParam<T>::getValue)
	.def("setValue", &eoValueParam<T>::setValue)
	//.add_property("value", getv<T>, setv<T>)
	;
}

void valueParam()
{
    class_<eoParam, ParamWrapper, boost::noncopyable>("eoParam", init<>())
	.def(init< std::string, std::string, std::string, char, bool>())
	.def("getValue", &ParamWrapper::getValue)
	.def("setValue", &ParamWrapper::setValue)
	.def("longName", &eoParam::longName, return_value_policy<copy_const_reference>())
	//.def("defValue", &eoParam::defValue, return_value_policy<copy_const_reference>())
	.def("description", &eoParam::description, return_value_policy<copy_const_reference>())
	.def("shortName", &eoParam::shortName)
	.def("required", &eoParam::required)
	;
    
    define_valueParam<int>("int");
    define_valueParam<std::vector<double> >("vec");
}

