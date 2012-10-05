#ifndef VALUEPARAM_H
#define VALUEPARAM_H

#include <string>
#include <boost/python.hpp>

class ValueParam : public eoParam // ValueParam containing python object
{
    boost::python::object obj;

public:

    ValueParam() : eoParam(), obj() {}

    ValueParam(boost::python::object o,
	       std::string longName,
	       std::string d = "No Description",
	       char s = 0,
	       bool r = false) : eoParam(longName, "", d, s, r)
    {
	std::cerr << "HI" << std::endl;
	obj = o;
	eoParam::defValue(getValue());
    }

    std::string getValue() const
    {
	boost::python::str s = boost::python::str(obj);
	return std::string(boost::python::extract<const char*>(s));
    }

    void setValue(const std::string& v)
    {
	std::cerr << "not implemented yet" << std::endl;
    }

    boost::python::object getObj() const { return obj;}
    void setObj(boost::python::object o) { obj = o; }
};

#endif
