/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#define _USE_MATH_DEFINES
#include <math.h>

#include <eo>
#include <edo>
#include <es.h>

typedef eoReal< eoMinimizingFitness > EOT;

int main(void)
{
    EOT sol;
    sol.push_back( M_PI * 1 );
    sol.push_back( M_PI * 2 );
    sol.push_back( M_PI * 3 );
    sol.push_back( M_PI * 4 );
    sol.push_back( M_PI * 4 + M_PI / 2 );
    sol.push_back( M_PI * 5 + M_PI / 2 );
    // we expect {pi,0,pi,0,pi/2,pi+pi/2}
    std::cout << "expect: INVALID  4 3.14159 0 3.14159 0 1.5708 4.71239" << std::endl;

    edoRepairer<EOT>* repare = new edoRepairerModulo<EOT>( 2 * M_PI ); // modulo 2pi

    (*repare)(sol);

    std::cout << sol << std::endl;

    return 0;
}
