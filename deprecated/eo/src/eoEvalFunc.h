// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEvalFunc.h
// (c) GeNeura Team, 1998
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
 */
//-----------------------------------------------------------------------------

#ifndef eoEvalFunc_H
#define eoEvalFunc_H

#include <eoFunctor.h>

/** @defgroup Evaluation Evaluation
 * @ingroup Operators
 */

/** Evaluate: takes one EO and sets its "fitness" property
 returning this fitness also. That is why EOT is passed by
 non-const reference: it must be altered within evaluate.\\

 The requirements on the types with which this class is to be
 instantiated with are null, or else, they depend on the particular
 class it's going to be applied to; EO does not impose any requirement
 on it. If you subclass this abstract class, and use it to evaluate an
 EO, the requirements on this EO will depend on the evaluator.

 @ingroup Evaluation
 @ingroup Core
 */
template<class EOT> class eoEvalFunc : public eoUF<EOT&, void>
{
    public :
        typedef EOT EOType;

        typedef typename EOT::Fitness EOFitT;
};

#endif
