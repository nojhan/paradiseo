// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoDistribution.h
// (c) Marc Schoenauer, Maarten Keijzer, 2001
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

    Contact: Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoDistribution_H
#define _eoDistribution_H

#include <algorithm>

#include <eoInit.h>
#include <eoPop.h>

/**
 * Abstract class for Distribution Evolution Algorithms within EO:
 *    the distribution itself
 *
 * It basically IS AN eoInit -  operator()(EOT &) generates new indis
 *
 * The instances will probably be eoValueParam of some kind
 *    see eoPBILDistrib.h
 *
 *  @ingroup Core
*/

template <class EOT>
class eoDistribution :  public eoInit<EOT>,
                        public eoPersistent, public eoObject
{
public:
  virtual void operator()(EOT &) = 0; // DO NOT FORGET TO INVALIDATE the EOT
};

#endif
