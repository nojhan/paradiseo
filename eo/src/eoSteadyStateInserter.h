/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoSteadyStateInserter.h
    Still abstract population insertion operator that is initialized with
    and eoEvalFunc object to be able to evaluate individuals before inserting
    them.

 (c) Maarten Keijzer (mak@dhi.dk) and GeNeura Team, 1999, 2000
 
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

#ifndef eoSteadyStateInserter_h
#define eoSteadyStateInserter_h


#include <eoEvalFunc.h>
#include <eoInserter.h>

/**
 * eoSteadyStateInserter: Interface class that enables an operator to update
 * a population with a new individual... it contains an eoEvalFunc object to 
 * make sure that every individual is evaluated before it is inserted
*/
template <class EOT>
class eoSteadyStateInserter : public eoPopInserter<EOT> 
{
    public :
        eoSteadyStateInserter(eoEvalFunc<EOT>& _eval) : eval(_eval) , eoPopInserter<EOT>() {}

    protected :

        eoEvalFunc<EOT>& eval;
};


#endif
