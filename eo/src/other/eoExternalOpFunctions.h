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
#include <eoRnd.h>
#include <eoEvalFunc.h>

/**
    Initialization of external struct, ctor expects a function of the following
    signature:

    External func();

  Where External is the user defined struct or class
*/
template <class F, class External>
class eoExternalInit : public eoRnd<eoExternalEO<F, External> >
{

public :

    typedef eoExternalEO<F, External> ExternalEO;
    
    eoExternalInit(External (*_init)(void)) : init(_init) {}


    ExternalEO operator()(void) { return (*init)(); }

private :

    External (*init)(void);
};

/**
    Evaluation of external struct, ctor expects a function of the following
    signature:

    Fit func(External&);

  Where External is the user defined struct or class and Fit the fitness type
*/
template <class F, class External>
class eoExternalEvalFunc : public eoEvalFunc<eoExternalEO<F, External> >
{
    public :

    typedef eoExternalEO<F, External> ExternalEO;

    eoExternalEvalFunc(F (*_eval)(const External&)) : eval(_eval) {}

    void operator()(ExternalEO& eo) const
    {
        eo.fitness( (*eval)(eo) );
    }

    private :

    F (*eval)(const External&);
};

/**
    Mutation of external struct, ctor expects a function of the following
    signature:

    void func(External&);

  Where External is the user defined struct or class
*/

template <class F, class External>
class eoExternalMonOp : public eoMonOp<eoExternalEO<F, External> >
{
    public :

    typedef eoExternalEO<F, External> ExternalEO;

    eoExternalMonOp(void (*_mutate)(External&)) : mutate(_mutate) {}

    void operator()(ExternalEO& eo) const
    {
        (*mutate)(eo);
        eo.invalidate();
    }

    private :

    void (*mutate)(External&);
};

/**
    Crossover of external struct, ctor expects a function of the following
    signature:

    void func(External&, const External&);

  Where External is the user defined struct or class
*/
template <class F, class External>
class eoExternalBinOp : public eoBinOp<eoExternalEO<F, External> >
{
    public :

    typedef eoExternalEO<F, External> ExternalEO;

    eoExternalBinOp(void (*_binop)(External&, const External&)) : binop(_binop) {}

    void operator()(ExternalEO& eo1, const ExternalEO& eo2) const
    {
        (*binop)(eo1, eo2);
        eo1.invalidate();
    }

    private :

    void (*binop)(External&, const External&);
};

/**
    Crossover of external struct, ctor expects a function of the following
    signature:

    void func(External&, External&);

  Where External is the user defined struct or class
*/
template <class F, class External>
class eoExternalQuadraticOp : public eoQuadraticOp<eoExternalEO<F, External> >
{
    public :

    typedef eoExternalEO<F, External> ExternalEO;

    eoExternalQuadraticOp(void (*_quadop)(External&, External&)) : quadop(_quadop) {}

    void operator()(ExternalEO& eo1, ExternalEO& eo2) const
    {
        (*quadop)(eo1, eo2);
        eo1.invalidate();
        eo2.invalidate();
    }

    private :

    void (*quadop)(External&, External&);
};



#endif
