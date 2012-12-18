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
    Pierre Savéant <pierre.saveant@thalesgroup.com>
*/

#include <eo>
#include <edo>
#include <es.h>

typedef eoReal< eoMinimizingFitness > EOT;

int main(void)
{
    EOT sol;
    sol.push_back(1.1);
    sol.push_back(1.1);
    sol.push_back(3.9);
    sol.push_back(3.9);
    sol.push_back(5.4);
    sol.push_back(5.6);
    sol.push_back(7.011);
    sol.push_back(8.09);
    sol.push_back(8.21);
    
    std::cout << "expect: INVALID  9 1 2 3 4 5 6 7 8.1 8.2" << std::endl;

    edoRepairer<EOT>* rep1 = new edoRepairerFloor<EOT>();
    edoRepairer<EOT>* rep2 = new edoRepairerCeil<EOT>();
    edoRepairer<EOT>* rep3 = new edoRepairerRound<EOT>();
    edoRepairer<EOT>* rep4 = new edoRepairerRoundDecimals<EOT>( 10 );

    std::vector<unsigned int> indexes1;
    indexes1.push_back(0);
    indexes1.push_back(2);

    std::vector<unsigned int> indexes2;
    indexes2.push_back(1);
    indexes2.push_back(3);

    std::vector<unsigned int> indexes3;
    indexes3.push_back(4);
    indexes3.push_back(5);

    std::vector<unsigned int> indexes4;
    indexes4.push_back(6);
    indexes4.push_back(7);
    indexes4.push_back(8);

    edoRepairerDispatcher<EOT> repare( indexes1, rep1 );
    repare.add( indexes2, rep2 );
    repare.add( indexes3, rep3 );
    repare.add( indexes4, rep4 );

    repare(sol);

    std::cout << sol << std::endl;

    return 0;
}
