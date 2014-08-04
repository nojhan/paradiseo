// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoInvalidateOps.h
// (c) Maarten Keijzer 2001
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
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoInvalidateOps_h
#define _eoInvalidateOps_h

#include <eoOp.h>

/** @addtogroup Utilities

One of the invalidator operators. Use this one as a 'hat' on an operator
that is defined to work on a generic datatype. This functor will then check
the return type of the operator and invalidate the fitness of the individual.

This functor is used in algorithms that work with straight eoMonOp, eoBinOp
or eoQuadOp operators, for instance eoSGA. Note that eoGenOp derived operators
generally do invalidate the fitness of the objects they have changed.

Return value means "Has_Changed" and not "Needs_To_Be_Invalidated"
as successive invalidation are not really a problem
*/

template <class EOT>
class eoInvalidateMonOp : public eoMonOp<EOT>
{
  public:
    eoInvalidateMonOp(eoMonOp<EOT>& _op) : op(_op) {}

    bool operator()(EOT& _eo)
    {
      if (op(_eo))
      {
        _eo.invalidate();
        return true;
      }

      return false;
    }

  private:
    eoMonOp<EOT>& op;
};

/**
One of the invalidator operators. Use this one as a 'hat' on an operator
that is defined to work on a generic datatype. This functor will then check
the return type of the operator and invalidate the fitness of the individual.

This functor is used in algorithms that work with straight eoMonOp, eoBinOp
or eoQuadOp operators, for instance eoSGA. Note that eoGenOp derived operators
generally do invalidate the fitness of the objects they have changed.

Return value means "Has_Changed" and not "Needs_To_Be_Invalidated"
as successive invalidation are not really a problem
*/

template <class EOT>
class eoInvalidateBinOp : public eoBinOp<EOT>
{
  public:
    eoInvalidateBinOp(eoBinOp<EOT>& _op) : op(_op) {}

    bool operator()(EOT& _eo, const EOT& _eo2)
    {
      if (op(_eo, _eo2))
      {
        _eo.invalidate();
        return true;
      }

      return false;
    }

  private:
    eoBinOp<EOT>& op;
};

/**
One of the invalidator operators. Use this one as a 'hat' on an operator
that is defined to work on a generic datatype. This functor will then check
the return type of the operator and invalidate the fitness of the individual.

This functor is used in algorithms that work with straight eoMonOp, eoBinOp
or eoQuadOp operators, for instance eoSGA. Note that eoGenOp derived operators
generally do invalidate the fitness of the objects they have changed.

Return value means "Has_Changed" and not "Needs_To_Be_Invalidated"
as successive invalidation are not really a problem
*/

template <class EOT>
class eoInvalidateQuadOp : public eoQuadOp<EOT>
{
  public:
    eoInvalidateQuadOp(eoQuadOp<EOT>& _op) : op(_op) {}

    bool operator()(EOT& _eo1, EOT& _eo2)
    {
      if (op(_eo1, _eo2))
      {
        _eo1.invalidate();
        _eo2.invalidate();
        return true;
      }
      return false;
    }

  private:
    eoQuadOp<EOT>& op;
};

#endif
