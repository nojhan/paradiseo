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
#include <stdexcept>

// Here's 'len'. Why? dunno
#include "valueParam.h"
#include <boost/python/detail/api_placeholder.hpp>

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
	return call_method<std::string>(self, "getValueAsString");
    }

    void setValue(const std::string& s)
    {
	call_method<void>(self, "setValueAsString", s);
    }
};

template <typename T>
struct ValueParam_pickle_suite : boost::python::pickle_suite
{
    static
    boost::python::tuple getstate(const eoValueParam<T>& _param)
    {
	str v(_param.getValue());
	str d(_param.description());
	str def(_param.defValue());
	str l(_param.longName());
	object s(_param.shortName());
	object r(_param.required());
	return make_tuple(v,d,def,l,s,r);
    }
    static
    void setstate(eoValueParam<T>& _param, boost::python::tuple pickled)
    {
	std::string v = extract<std::string>(pickled[0]);
	std::string d = extract<std::string>(pickled[1]);
	std::string def = extract<std::string>(pickled[2]);
	std::string l = extract<std::string>(pickled[3]);
	char s = extract<char>(pickled[4]);
	bool r = extract<bool>(pickled[5]);

	_param = eoValueParam<T>(T(), l, d, s, r);
	_param.defValue(d);
	_param.setValue(v);
    }
};

template <class T, class U>
U getv(const eoValueParam<T>& v)    { return v.value(); }

template <class T, class U>
void setv(eoValueParam<T>& v, U val) { v.value() = val; }

template <>
numeric::array getv< std::vector<double>, numeric::array >
(const eoValueParam< std::vector<double> >& param)
{
    const std::vector<double>& v = param.value();
    list result;

    for (unsigned i =0; i < v.size(); ++i)
	result.append(v[i]);

    return numeric::array(result);
}

template <>
void setv< std::vector<double>, numeric::array >
(eoValueParam< std::vector<double> >& param, numeric::array val)
{
    std::vector<double>& v = param.value();
    v.resize( boost::python::len(val) );
    for (unsigned i = 0; i < v.size(); ++i)
	{
	    extract<double> x(val[i]);
	    if (!x.check())
		throw std::runtime_error("double expected");

	    v[i] = x();
	}
}

template <>
tuple getv<std::pair<double, double>, tuple >
    (const eoValueParam< std::pair<double,double> >& p)
{
    return make_tuple(p.value().first, p.value().second);
}

template <>
void setv< std::pair<double, double>, tuple >
(eoValueParam< std::pair<double,double> >& p, tuple val)
{
    extract<double> first(val[0]);
    extract<double> second(val[1]);

    if (!first.check())
	throw std::runtime_error("doubles expected");
    if (!second.check())
	throw std::runtime_error("doubles expected");

    p.value().first = first();
    p.value().second = second();
}

template <class T, class U>
void define_valueParam(std::string prefix)
{
    std::string name = "eoValueParam";
    name += prefix;

    class_<eoValueParam<T>, bases<eoParam> >(name.c_str(), init<>())
	.def(init<T, std::string, std::string, char, bool>())
	.def(init<T, std::string, std::string, char>())
	.def(init<T, std::string, std::string>())
	.def(init<T, std::string>())
	.def("getValueAsString", &eoValueParam<T>::getValue)
	.def("__str__", &eoValueParam<T>::getValue)
	.def("setValueAsString", &eoValueParam<T>::setValue)
	.def("getValue", getv<T, U>)
	.def("setValue", setv<T, U>)
	.add_property("value", getv<T, U>, setv<T, U>)
	.def_pickle(ValueParam_pickle_suite<T>())
	;
}

void valueParam()
{
    class_<eoParam, ParamWrapper, boost::noncopyable>("eoParam", init<>())
	.def(init< std::string, std::string, std::string, char, bool>())
	.def("getValueAsString", &ParamWrapper::getValue)
	.def("setValueAsString", &ParamWrapper::setValue)
	.def("longName", &eoParam::longName, return_value_policy<copy_const_reference>())
	//.def("defValue", &eoParam::defValue, return_value_policy<copy_const_reference>())
	.def("description", &eoParam::description, return_value_policy<copy_const_reference>())
	.def("shortName", &eoParam::shortName)
	.def("required", &eoParam::required)
	;

    define_valueParam<int, int>("Int");
    define_valueParam<double, double>("Float");
    define_valueParam<std::vector<double>, numeric::array >("Vec");
    define_valueParam< std::pair<double, double>, tuple >("Pair");
    //define_valueParam< object, object >("Py");

    class_<ValueParam, bases<eoParam> >("eoValueParam", init<>())
	//.def(init<object, std::string, std::string, char, bool>())
	//.def(init<object, std::string, std::string, char>())
	//.def(init<object, std::string, std::string>())
	.def(init<object, std::string>())
	.def("getValueAsString", &ValueParam::getValue)
	.def("__str__", &ValueParam::getValue)
	.def("setValueAsString", &ValueParam::setValue)
	.add_property("object", &ValueParam::getObj, &ValueParam::setObj)
	;
}
