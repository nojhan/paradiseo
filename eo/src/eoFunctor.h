// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFunctor.h
// (c) Maarten Keijzer 2000
/* 
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoFunctor_h
#define _eoFunctor_h

/// Base class for functors to get a nice hierarchy diagram
class eoFunctorBase 
{
public :
    virtual ~eoFunctorBase() {}

    /// tag to identify a procedure in compile time function selection @see functor_category
    struct procedure_tag {};
    /// tag to identify a unary function in compile time function selection @see functor_category
    struct unary_function_tag {};
    /// tag to identify a binary function in compile time function selection @see functor_category
    struct binary_function_tag {};
};

/** 
    Basic Procedure. Derive from this class when defining 
    any procedure. It defines a result_type that can be used
    to determine the return type
    Argument and result types can be any type including void for
    result_type
**/
template <class R>
class eoProcedure : public eoFunctorBase
{
public :
    
    /// virtual dtor here so there is no need to define it in derived classes
    virtual ~eoProcedure() {}

    typedef R result_type;
    
    /// The pure virtual function that needs to be implemented by the subclass
    virtual R operator()() = 0;
};

/**
    Overloaded function that can help in the compile time detection 
    of the type of functor we are dealing with

    @see eoCounter, make_counter
*/
template<class R>
eoFunctorBase::procedure_tag functor_category(const eoProcedure<R>&)
{
    return eoFunctorBase::procedure_tag();
}

/** 
    Basic Unary Functor. Derive from this class when defining 
    any unary function. First template argument is result_type, second
    is first_argument_type.
    Argument and result types can be any type including void for
    result_type
**/
template <class R, class A1>
class eoUnaryFunctor : public eoFunctorBase
{
public :

    /// virtual dtor here so there is no need to define it in derived classes
    virtual ~eoUnaryFunctor() {}

    typedef R result_type;
    typedef A1 first_argument_type;

    /// The pure virtual function that needs to be implemented by the subclass
    virtual R operator()(A1) = 0;
};

/**
    Overloaded function that can help in the compile time detection 
    of the type of functor we are dealing with
    @see eoCounter, make_counter
*/
template<class R, class A1>
eoFunctorBase::unary_function_tag functor_category(const eoUnaryFunctor<R, A1>&)
{
    return eoFunctorBase::unary_function_tag();
}


/** 
    Basic Binary Functor. Derive from this class when defining 
    any binary function. First template argument is result_type, second
    is first_argument_type, third is second_argument_type.
    Argument and result types can be any type including void for
    result_type
**/
template <class R, class A1, class A2>
class eoBinaryFunctor : public eoFunctorBase
{
public :
        /// virtual dtor here so there is no need to define it in derived classes
    virtual ~eoBinaryFunctor() {}
    
    typedef R result_type;
    typedef A1 first_argument_type;
    typedef A2 second_argument_type;

    /// The pure virtual function that needs to be implemented by the subclass
    virtual R operator()(A1, A2) = 0;
};

/**
    Overloaded function that can help in the compile time detection 
    of the type of functor we are dealing with
    @see eoCounter, make_counter
*/
template<class R, class A1, class A2>
eoFunctorBase::binary_function_tag functor_category(const eoBinaryFunctor<R, A1, A2>&)
{
    return eoFunctorBase::binary_function_tag();
}


#endif