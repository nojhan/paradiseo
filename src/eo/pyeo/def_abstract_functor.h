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

#ifndef MAKE_ABSTRACT_FUNCTOR_H
#define MAKE_ABSTRACT_FUNCTOR_H

#include "../eoFunctor.h"

// DEFINES for call
#define WC1 boost::python::with_custodian_and_ward<1,2>()
#define WC2 boost::python::with_custodian_and_ward<1,2, with_custodian_and_ward<1,3> >()

namespace eoutils {

    using namespace boost::python;

    template <class Proc>
    class ProcWrapper : public Proc
    {
    public:
        PyObject* self;
        ProcWrapper(PyObject* s) : self(s) {}

        typename Proc::result_type operator()(void)
        {
            return boost::python::call_method<typename Proc::result_type>(self, "__call__");
        }
    };

    template <class Proc>
    void make_abstract_functor(std::string name, typename eoFunctorBase::procedure_tag)
    {
        typedef ProcWrapper<Proc> Wrapper;
        boost::python::class_<Proc, Wrapper,boost::noncopyable>(name.c_str(), boost::python::init<>() )
            .def("__call__", &Wrapper::operator());
    }

    template <class Proc>
    void make_abstract_functor_ref(std::string name, typename eoFunctorBase::procedure_tag)
    {
        typedef ProcWrapper<Proc> Wrapper;
        boost::python::class_<Proc, Wrapper,boost::noncopyable>(name.c_str(), boost::python::init<>() )
            .def("__call__", &Wrapper::operator(), boost::python::return_internal_reference<>());
    }

    template <class Unary>
    class UnaryWrapper : public Unary
    {
    public:
        PyObject* self;
        UnaryWrapper(PyObject* s) : self(s) {}

        typename Unary::result_type operator()(typename Unary::argument_type a)
        {
            return boost::python::call_method<typename Unary::result_type>(self, "__call__", boost::ref(a) );
        }
    };

    template <class Unary>
    void make_abstract_functor(std::string name, typename eoFunctorBase::unary_function_tag)
    {
        typedef UnaryWrapper<Unary> Wrapper;

        boost::python::class_<Unary, Wrapper, boost::noncopyable>(name.c_str(), boost::python::init<>() )
            .def("__call__", &Wrapper::operator())
            ;
    }

    template <class Unary>
    void make_abstract_functor_ref(std::string name, typename eoFunctorBase::unary_function_tag)
    {
        typedef UnaryWrapper<Unary> Wrapper;

        boost::python::class_<Unary, Wrapper, boost::noncopyable>(name.c_str(), boost::python::init<>() )
            .def("__call__", &Wrapper::operator(), boost::python::return_internal_reference<>() )
            ;
    }

    template <class Binary>
    class BinaryWrapper : public Binary
    {
    public:
        PyObject* self;
        BinaryWrapper(PyObject* s) : self(s) {}
        typename Binary::result_type operator()(typename Binary::first_argument_type a1, typename Binary::second_argument_type a2)
        {
            return boost::python::call_method<
            typename Binary::result_type>(self, "__call__", boost::ref(a1), boost::ref(a2) );
        }
    };

    template <class Binary>
    void make_abstract_functor(std::string name, typename eoFunctorBase::binary_function_tag)
    {
        typedef BinaryWrapper<Binary> Wrapper;
        boost::python::class_<Binary, Wrapper, boost::noncopyable>(name.c_str(), boost::python::init<>() )
            .def("__call__", &Wrapper::operator());
    }

    template <class Binary>
    void make_abstract_functor_ref(std::string name, typename eoFunctorBase::binary_function_tag)
    {
        typedef BinaryWrapper<Binary> Wrapper;
        boost::python::class_<Binary, Wrapper, boost::noncopyable>(name.c_str(), boost::python::init<>() )
            .def("__call__", &Wrapper::operator(), boost::python::return_internal_reference<>() );
    }

}// namespace eoutils

template <class Functor>
void def_abstract_functor(std::string name)
{
    eoutils::make_abstract_functor<Functor>(name, Functor::functor_category());
}

template <class Functor>
void def_abstract_functor_ref(std::string name)
{
    eoutils::make_abstract_functor_ref<Functor>(name, Functor::functor_category());
}

#endif
