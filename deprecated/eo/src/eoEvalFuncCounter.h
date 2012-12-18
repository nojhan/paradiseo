// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEvalFuncCounter.h
// (c) Maarten Keijzer, Marc Schoenauer and GeNeura Team, 2000
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
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoEvalFuncCounter_H
#define eoEvalFuncCounter_H

#include <eoEvalFunc.h>
#include <utils/eoParam.h>

/**
Counts the number of evaluations actually performed.

@ingroup Evaluation
*/
template<class EOT> class eoEvalFuncCounter : public eoEvalFunc<EOT>, public eoValueParam<unsigned long>
{
    public :
        eoEvalFuncCounter(eoEvalFunc<EOT>& _func, std::string _name = "Eval. ")
            : eoValueParam<unsigned long>(0, _name), func(_func) {}

        virtual void operator()(EOT& _eo)
        {
            if (_eo.invalid())
            {
                value()++;
                func(_eo);
            }
        }

    protected :
        eoEvalFunc<EOT>& func;
};

#endif
