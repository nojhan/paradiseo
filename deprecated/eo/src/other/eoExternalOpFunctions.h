/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoExternalOpFunc.h
        Defines eoExternalInitOpFunc, eoExternalMonOpFunc, eoExternalBinOpFunc, eoExternalQuadOpFunc
        that are used to wrap a function pointer to externally defined initialization
        and 'genetic' operators

 (c) Maarten Keijzer (mkeijzer@mad.scientist.com) and GeNeura Team, 1999, 2000

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
 */

#ifndef eoExternalOpFunc_h
#define eoExternalOpFunc_h

#include <other/eoExternalEO.h>
#include <eoOp.h>
#include <eoInit.h>
#include <eoEvalFunc.h>

/**
    Initialization of external struct, ctor expects a function of the following
    signature:

    External func();

  Where External is the user defined struct or class

  @ingroup Utilities
*/
template <class F, class External, class ExternalEO = eoExternalEO<F, External> >
class eoExternalInit : public eoInit<ExternalEO>
{

public :

    eoExternalInit(External (*_init)(void)) : init(_init) {}


    void operator()(ExternalEO& _eo)
    {
        _eo.External::operator=( (*init)() );
        _eo.invalidate();
    }

private :

    External (*init)(void);
};

/**
    Evaluation of external struct, ctor expects a function of the following
    signature:

    Fit func(External&);

  Where External is the user defined struct or class and Fit the fitness type

@ingroup Utilities
*/
template <class F, class External, class ExternalEO = eoExternalEO<F, External> >
class eoExternalEvalFunc : public eoEvalFunc<ExternalEO>
{
    public :

    eoExternalEvalFunc(F (*_eval)(const External&)) : eval(_eval) {}

    void operator()(ExternalEO& eo)
    {
        if (eo.invalid())
            eo.fitness( (*eval)(eo) );
    }

    private :

    F (*eval)(const External&);
};

/**
    Mutation of external struct, ctor expects a function of the following
    signature:

    bool func(External&);


  Where External is the user defined struct or class.
  The function should return true when it changed something, false otherwise

  @ingroup Utilities
*/

template <class F, class External, class ExternalEO = eoExternalEO<F, External> >
class eoExternalMonOp : public eoMonOp<ExternalEO>
{
    public :

    eoExternalMonOp(bool (*_mutate)(External&)) : mutate(_mutate) {}

    bool operator()(ExternalEO& eo)
    {
        return (*mutate)(eo);
    }

    private :

    bool (*mutate)(External&);
};

/**
    Crossover of external struct, ctor expects a function of the following
    signature:

    bool func(External&, const External&);

  Where External is the user defined struct or class
  The function should return true when it changed something, false otherwise

  @ingroup Utilities
*/
template <class F, class External, class ExternalEO = eoExternalEO<F, External> >
class eoExternalBinOp : public eoBinOp<ExternalEO>
{
    public :

    eoExternalBinOp(bool (*_binop)(External&, const External&)) : binop(_binop) {}

    bool operator()(ExternalEO& eo1, const ExternalEO& eo2)
    {
        return (*binop)(eo1, eo2);
    }

    private :

    bool (*binop)(External&, const External&);
};

/**
    Crossover of external struct, ctor expects a function of the following
    signature:

    bool func(External&, External&);

  Where External is the user defined struct or class
  The function should return true when it changed something, false otherwise

  @ingroup Utilities
*/
template <class F, class External, class ExternalEO = eoExternalEO<F, External> >
class eoExternalQuadOp : public eoQuadOp<ExternalEO>
{
    public :

    eoExternalQuadOp(bool (*_quadop)(External&, External&)) : quadop(_quadop) {}

    bool operator()(ExternalEO& eo1, ExternalEO& eo2)
    {
        return (*quadop)(eo1, eo2);
    }

    private :

    bool (*quadop)(External&, External&);
};

#endif
