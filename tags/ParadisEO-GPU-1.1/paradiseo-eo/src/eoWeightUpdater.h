// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoWeightUpdater.h
// (c) OPAC 2007
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

    Contact: thomas.legrand@lifl.fr
 */
//-----------------------------------------------------------------------------

#ifndef EOWEIGHTUPDATER_H
#define EOWEIGHTUPDATER_H

//-----------------------------------------------------------------------------
#include <eoFunctor.h>
//-----------------------------------------------------------------------------

/**
 * Abstract class for (inertia) weight updater (particle swarm optimization).
 * Used inside classes extending eoVelocity.
 *
 * @ingroup Core
 */
template < class WeightType > class eoWeightUpdater: public eoUF < WeightType &, void > {};


#endif/*EOWEIGHTUPDATER_H*/
