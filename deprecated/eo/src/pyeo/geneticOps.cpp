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

#include <eoGenOp.h>
#include <eoOp.h>
#include <eoCloneOps.h>
#include <eoPopulator.h>
#include <eoOpContainer.h>

#include "PyEO.h"
#include "def_abstract_functor.h"

using namespace boost::python;

class GenOpWrapper : public eoGenOp<PyEO>
{
public:

    PyObject* self;
    GenOpWrapper(PyObject* p) : self(p) {}
    unsigned max_production(void)
    {
        return call_method<unsigned>(self,"max_production");
    }
    std::string className() const
    {
        return "GenOpDerivative"; // never saw the use of className anyway
    }

    void apply(eoPopulator<PyEO>& populator )
    {
        boost::python::call_method<void>(self,"apply", boost::ref( populator ) );
    }
};

class PopulatorWrapper : public eoPopulator<PyEO>
{
public:
    PyObject* self;
    PopulatorWrapper(PyObject* p, const eoPop<PyEO>& src, eoPop<PyEO>& dest)
        : eoPopulator<PyEO>(src, dest), self(p)
    {
        //throw std::runtime_error("abstract base class");
    }

    const PyEO& select()
    {
        return call_method<const PyEO&>(self,"select");
    }
};

class MonOpWrapper : public eoMonOp<PyEO>
{
public:
    PyObject* self;
    MonOpWrapper(PyObject* p) : self(p) {}
    bool operator()(PyEO& _eo)
    { return boost::python::call_method<bool>(self, "__call__", boost::ref( _eo )); }
};
class BinOpWrapper : public eoBinOp<PyEO>
{
public:
    PyObject* self;
    BinOpWrapper(PyObject* p) : self(p) {}
    bool operator()(PyEO& _eo, const PyEO& _eo2)
    { return boost::python::call_method<bool>(self, "__call__", boost::ref( _eo ), boost::ref(_eo2)); }
};
class QuadOpWrapper : public eoQuadOp<PyEO>
{
public:
    PyObject* self;
    QuadOpWrapper(PyObject* p) : self(p) {}
    bool operator()(PyEO& _eo, PyEO& _eo2)
    { return boost::python::call_method<bool>(self, "__call__", boost::ref( _eo ), boost::ref(_eo2)); }
};

void geneticOps()
{
    class_<eoPopulator<PyEO>, PopulatorWrapper, boost::noncopyable>
        ("eoPopulator", init<const eoPop<PyEO>&, eoPop<PyEO>&>() )
        .def("select", &PopulatorWrapper::select,  return_internal_reference<>() )
        .def("get", &eoPopulator<PyEO>::operator*, return_internal_reference<>() )
        .def("next", &eoPopulator<PyEO>::operator++, return_internal_reference<>() )
        .def("insert", &eoPopulator<PyEO>::insert)
        .def("reserve", &eoPopulator<PyEO>::reserve)
        .def("source", &eoPopulator<PyEO>::source, return_internal_reference<>() )
        .def("offspring", &eoPopulator<PyEO>::offspring, return_internal_reference<>() )
        .def("tellp", &eoPopulator<PyEO>::tellp)
        .def("seekp", &eoPopulator<PyEO>::seekp)
        .def("exhausted", &eoPopulator<PyEO>::exhausted)
        ;

    class_<eoSeqPopulator<PyEO>, bases<eoPopulator<PyEO> > >
        ("eoSeqPopulator", init<const eoPop<PyEO>&, eoPop<PyEO>&>() )
        .def("select", &eoSeqPopulator<PyEO>::select, return_internal_reference<>() )
        ;

    class_<eoSelectivePopulator<PyEO>, bases<eoPopulator<PyEO> > >
        ("eoSelectivePopulator", init<const eoPop<PyEO>&, eoPop<PyEO>&, eoSelectOne<PyEO>& >() )
        .def("select", &eoSeqPopulator<PyEO>::select, return_internal_reference<>() )
        ;
    enum_<eoOp<PyEO>::OpType>("OpType")
        .value("unary", eoOp<PyEO>::unary)
        .value("binary", eoOp<PyEO>::binary)
        .value("quadratic", eoOp<PyEO>::quadratic)
        .value("general", eoOp<PyEO>::general)
        ;

    class_<eoOp<PyEO> >("eoOp", init<eoOp<PyEO>::OpType>())
        .def("getType", &eoOp<PyEO>::getType);

    class_<eoMonOp<PyEO>, MonOpWrapper, bases<eoOp<PyEO> >, boost::noncopyable>("eoMonOp", init<>())
        .def("__call__", &MonOpWrapper::operator(), "an example docstring");
    class_<eoBinOp<PyEO>, BinOpWrapper, bases<eoOp<PyEO> >, boost::noncopyable>("eoBinOp", init<>())
        .def("__call__", &BinOpWrapper::operator());
    class_<eoQuadOp<PyEO>, QuadOpWrapper, bases<eoOp<PyEO> >, boost::noncopyable>("eoQuadOp", init<>())
        .def("__call__", &QuadOpWrapper::operator());

    class_<eoGenOp<PyEO>, GenOpWrapper, bases<eoOp<PyEO> >, boost::noncopyable>("eoGenOp", init<>())
        .def("max_production", &GenOpWrapper::max_production)
        .def("className", &GenOpWrapper::className)
        .def("apply", &GenOpWrapper::apply)
        .def("__call__", &eoGenOp<PyEO>::operator())
        ;

    class_<eoSequentialOp<PyEO>, bases<eoGenOp<PyEO> >, boost::noncopyable>("eoSequentialOp", init<>())
        .def("add", &eoSequentialOp<PyEO>::add, WC1)
        .def("apply", &eoSequentialOp<PyEO>::apply)
        ;

    class_<eoProportionalOp<PyEO>, bases<eoGenOp<PyEO> >, boost::noncopyable>("eoProportionalOp", init<>())
        .def("add", &eoProportionalOp<PyEO>::add, WC1)
        .def("apply", &eoProportionalOp<PyEO>::apply)
        ;

    /* Cloning */
    class_<eoMonCloneOp<PyEO>, bases<eoMonOp<PyEO> > >("eoMonCloneOp").def("__call__", &eoMonCloneOp<PyEO>::operator());
    class_<eoBinCloneOp<PyEO>, bases<eoBinOp<PyEO> > >("eoBinCloneOp").def("__call__", &eoBinCloneOp<PyEO>::operator());
    class_<eoQuadCloneOp<PyEO>, bases<eoQuadOp<PyEO> > >("eoQuadCloneOp").def("__call__", &eoQuadCloneOp<PyEO>::operator());

}
