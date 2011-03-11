#ifndef _FUNCTORS_h
#define _FUNCTORS_h

#include <functional>

/** Base class for functors to get a nice hierarchy diagram

    That's actually quite an understatement as it does quite a bit more than
    just that. By having all functors derive from the same base class, we can
    do some memory management that would otherwise be very hard.

    The memory management base class is called eoFunctorStore, and it supports
    a member add() to add a pointer to a functor. When the functorStore is
    destroyed, it will delete all those pointers. So beware: do not delete
    the functorStore before you are done with anything that might have been allocated.

    @see eoFunctorStore

*/
class FunctorBase
{
public :
    /// virtual dtor here so there is no need to define it in derived classes
    virtual ~FunctorBase() {}
};

/**
    Basic Function. Derive from this class when defining
    any procedure. It defines a result_type that can be used
    to determine the return type
    Argument and result types can be any type including void for
    result_type
**/
template <class R>
class Functor : public FunctorBase
{
public :

    /// virtual dtor here so there is no need to define it in derived classes
    virtual ~Functor() {}

  /// the return type - probably useless ....
    typedef R result_type;

    /// The pure virtual function that needs to be implemented by the subclass
    virtual R operator()() = 0;

};

 
/** 
    Basic Unary Functor. Derive from this class when defining 
    any unary function. First template argument is the first_argument_type,
    second result_type.
    Argument and result types can be any type including void for
    result_type
**/
template <class A1, class R>
class FunctorUnary : public FunctorBase, public std::unary_function<A1, R>
{
public :

    /// virtual dtor here so there is no need to define it in derived classes
    virtual ~FunctorUnary() {}

    /// The pure virtual function that needs to be implemented by the subclass
    virtual R operator()(A1) = 0;

};


/** 
    Basic Binary Functor. Derive from this class when defining 
    any binary function. First template argument is result_type, second
    is first_argument_type, third is second_argument_type.
    Argument and result types can be any type including void for
    result_type
**/
template <class A1, class A2, class R>
class FunctorBinary : public FunctorBase, public std::binary_function<A1, A2, R>
{
public :
        /// virtual dtor here so there is no need to define it in derived classes
    virtual ~FunctorBinary() {}
    
    //typedef R result_type;
    //typedef A1 first_argument_type;
    //typedef A2 second_argument_type;

    /// The pure virtual function that needs to be implemented by the subclass
    virtual R operator()(A1, A2) = 0;

};



#endif
