// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoCounter.h
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

#ifndef _eoCounter_h
#define _eoCounter_h

#include <eoFunctor.h>
#include <eoFunctorStore.h>
#include <utils/eoParam.h>

/**
        Generic counter class that counts the number of times
        a procedure is used. Add a procedure through its ctor and
        use this class instead of it.

    It is derived from eoValueParam so you can add it to a monitor.

    @ingroup Utilities
*/
template <class Procedure>
class eoProcedureCounter : public Procedure, public eoValueParam<unsigned long>
{
    public:

        eoProcedureCounter(Procedure& _proc, std::string _name = "proc_counter")
            : eoValueParam<unsigned long>(0, _name), proc(_proc) {}

        /** Calls the embedded function and increments the counter

          Note for MSVC users, if this code does not compile, you are quite
          likely trying to count a function that has a non-void return type.
          Don't look at us, look at the MSVC builders. Code like "return void;"
          is perfectly legal according to the ANSI standard, but the guys at
          Microsoft didn't get to implementing it yet.

          We had two choices: assuming (and compiling ) code that returns void or code that returns non-void.
          Given that in EO most functors return void, it was chosen to support void.

          But also please let me know if you have a compiler that defines _MSC_VER (lot's of windows compilers do), but is quite
          capable of compiling return void; type of code. We'll try to change the signature then.

          You happy GNU (and other compiler) users will not have a problem with this..
        */
        typename Procedure::result_type operator()(void)
        {
            value()++;
#ifdef _MSC_VER
            proc();
#else
            return proc();
#endif
        }

    private :

        Procedure& proc;
};

/**
        Generic counter class that counts the number of times
        a unary function is used. Add a unary function through its ctor and
        use this class instead of it.

    It is derived from eoValueParam so you can add it to a monitor.

    Example: suppose you have an eoEvalFunc called myeval, to count the
    number of evaluations, just define:

  eoUnaryFunctorCounter<void, EoType> evalCounter(myeval);

  and use evalCounter now instead of myeval.
*/

template <class UnaryFunctor>
class eoUnaryFunctorCounter : public UnaryFunctor, public eoValueParam<unsigned long>
{
    public:
        eoUnaryFunctorCounter(UnaryFunctor& _func, std::string _name = "uf_counter")
            : eoValueParam<unsigned long>(0, _name), func(_func) {}

        /** Calls the embedded function and increments the counter

          Note for MSVC users, if this code does not compile, you are quite
          likely trying to count a function that has a non-void return type.
          Don't look at us, look at the MSVC builders. Code like "return void;"
          is perfectly legal according to the ANSI standard, but the guys at
          Microsoft didn't get to implementing it yet.

          We had two choices: assuming (and compiling ) code that returns void or code that returns non-void.
          Given that in EO most functors return void, it was chosen to support void.

          But also please let me know if you have a compiler that defines _MSC_VER (lot's of windows compilers do), but is quite
          capable of compiling return void; type of code. We'll try to change the signature then.

          You happy GNU (and other compiler) users will not have a problem with this.
        */
        typename UnaryFunctor::result_type operator()
            (typename UnaryFunctor::first_argument_type _arg1)
        {
            value()++;
#ifdef _MSC_VER
            func(_arg1);
#else
            return func(_arg1);
#endif
        }

    private :

        UnaryFunctor& func;
};

/**
        Generic counter class that counts the number of times
        a binary function is used. Add a binary function through its ctor and
        use this class instead of it.

    It is derived from eoValueParam so you can add it to a monitor.

*/
template <class BinaryFunctor>
class eoBinaryFunctorCounter : public BinaryFunctor, public eoValueParam<unsigned long>
{
    public:
        eoBinaryFunctorCounter(BinaryFunctor& _func, std::string _name = "proc_counter")
            : eoValueParam<unsigned long>(0, _name), func(_func) {}

        /** Calls the embedded function and increments the counter

          Note for MSVC users, if this code does not compile, you are quite
          likely trying to count a function that has a non-void return type.
          Don't look at us, look at the MSVC builders. Code like "return void;"
          is perfectly legal according to the ANSI standard, but the guys at
          Microsoft didn't get to implementing it yet.

          We had two choices: assuming (and compiling ) code that returns void or code that returns non-void.
          Given that in EO most functors return void, it was chosen to support void.


          But also please let me know if you have a compiler that defines _MSC_VER (lot's of windows compilers do), but is quite
          capable of compiling return void; type of code. We'll try to change the signature then.

          You happy GNU (and other compiler) users will not have a problem with this.
        */
        typename BinaryFunctor::result_type operator()
            (typename BinaryFunctor::first_argument_type _arg1,
             typename BinaryFunctor::second_argument_type _arg2)
        {
            value()++;
#ifdef _MSC_VER
            func(_arg1, _arg2);
#else
            return func(_arg1, _arg2);
#endif
  }

    private :

        BinaryFunctor& func;
};

/** make_counter: overloaded function to make a counter out of a function

    how it works...

    by using the xxx_function_tag structure defined in eoFunctionBase, you
    can easily create a counter from a general class (say eoEval<EOT>), by
    simply stating:

    eoEval<EOT>& myCounted = make_counter(functor_category(myEval), myEval, store)

    @see eoFunctorBase, functor_category, eoFunctorStore

*/
template <class Procedure>
eoProcedureCounter<Procedure>& make_counter(eoFunctorBase::procedure_tag, Procedure& _proc, eoFunctorStore& store, std::string _name = "proc_counter")
{
    eoProcedureCounter<Procedure>* result = new eoProcedureCounter<Procedure>(_proc, _name);
    store.storeFunctor(result);
    return *result;
}

template <class UnaryFunctor>
eoUnaryFunctorCounter<UnaryFunctor>& make_counter(eoFunctorBase::unary_function_tag, UnaryFunctor& _proc, eoFunctorStore& store, std::string _name = "uf_counter")
{
    eoUnaryFunctorCounter<UnaryFunctor>* result = new eoUnaryFunctorCounter<UnaryFunctor>(_proc, _name);
    store.storeFunctor(result);
    return *result;
}

template <class BinaryFunctor>
eoBinaryFunctorCounter<BinaryFunctor>& make_counter(eoFunctorBase::binary_function_tag, BinaryFunctor& _proc, eoFunctorStore& store, std::string _name = "uf_counter")
{
    eoBinaryFunctorCounter<BinaryFunctor>* result = new eoBinaryFunctorCounter<BinaryFunctor>(_proc, _name);
    store.storeFunctor(result);
    return *result;
}

#endif
