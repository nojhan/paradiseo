// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSIGContinue.cpp
// (c) EEAAX 1996 - GeNeura Team, 1998 - Maarten Keijzer 2000
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
             Johann Dréo <nojhan@gmail.com>
             Caner Candan <caner@candan.fr>
*/
//-----------------------------------------------------------------------------
#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#include <signal.h>
#include <iostream>

// --- Global variables - but don't know what else to do - MS ---
bool     existSIGContinue = false;
bool     call_func = false;
