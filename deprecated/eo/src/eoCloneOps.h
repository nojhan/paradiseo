// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-
//-----------------------------------------------------------------------------
// eoCloneOps.h
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
    CVS Info: $Date: 2003-02-27 19:26:09 $ $Header: /home/nojhan/dev/eodev/eodev_cvs/eo/src/eoCloneOps.h,v 1.2 2003-02-27 19:26:09 okoenig Exp $ $Author: okoenig $
 */
//-----------------------------------------------------------------------------

#ifndef _eoCloneOps_H
#define _eoCloneOps_H

#include <eoOp.h>

/**
 * The different null-variation operators (i.e. they do nothing)
 *
 * eoQuadCloneOp at least is useful to emulate the standard
 *               crossover(pCross) + mutation(pMut)
 *               within the eoGenOp framework
 * eoMonCloneOp will probably be useful as the copy operator
 * eoBinCloneOp will certainly never been used - but let's be complete :-)
 *
 * @addtogroup Core
 * @{
 */

/**
Mon clone: one argument
*/
template <class EOT>
class eoMonCloneOp: public eoMonOp<EOT>
{
public:
  /// Ctor
  eoMonCloneOp() : eoMonOp<EOT>() {}
  virtual std::string className() const {return "eoMonCloneOp";}
  virtual bool operator()(EOT&){return false;}
};


/** Binary clone: two operands, only the first could be modified
 */
template<class EOT>
class eoBinCloneOp: public eoBinOp<EOT>
{
public:
  /// Ctor
  eoBinCloneOp() : eoBinOp<EOT>() {}
  virtual std::string className() const {return "eoBinCloneOp";}
  virtual bool operator()(EOT&, const EOT&){return false;}
};

/** Quad clone: two operands, both could be modified - but are not!
*/
template<class EOT>
class eoQuadCloneOp: public eoQuadOp<EOT>
{
public:
  /// Ctor
  eoQuadCloneOp():eoQuadOp<EOT>() {}
  virtual std::string className() const {return "eoQuadCloneOp";}
virtual bool operator()(EOT& , EOT& ) {return false;}
};

#endif
/** @} */
