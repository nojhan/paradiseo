// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// es.h
// (c) GeNeura Team 1998 - Maarten Keijzer 2000 - Marc Schoenauer 2001
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
             mak@dhi.dk
             todos@geneura.ugr.es, http://geneura.ugr.es
 */
//-----------------------------------------------------------------------------

#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#ifndef _es_h
#define _es_h

// contains the include specific to real representations, i.e. in src/es dir
//-----------------------------------------------------------------------------

// the genotypes - from plain std::vector<double> to full correlated mutation
#include <es/eoReal.h>
#include <es/eoEsSimple.h>
#include <es/eoEsStdev.h>
#include <es/eoEsFull.h>

// the initialization
#include <es/eoEsChromInit.h>

// general operators
#include <es/eoRealOp.h>
#include <es/eoNormalMutation.h>
#include <es/eoRealAtomXover.h>	// for generic operators

// SBX crossover (following Deb)
#include <es/eoSBXcross.h>

// ES specific operators
#include <es/eoEsGlobalXover.h> // Global ES Xover
#include <es/eoEsStandardXover.h> // 2-parents ES Xover

// the ES-mutations
#include <es/eoEsMutationInit.h>
#include <es/eoEsMutate.h>

#endif
