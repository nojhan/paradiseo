/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

-----------------------------------------------------------------------------
    eoException.h
      Exceptions that are possibly thrown at initialization and such should be
      defined here.

    (c) GeNeura Team, 2000
 
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

#ifndef eoException_h
#define eoException_h

#include <exception>

#include "eoObject.h"

struct eoException
{
    eoException() {}
    eoException(const eoObject& caller) : who_caught_it(caller.className()) {}

    std::string who(void) const { return who_caught_it; }
    virtual std::string what(void) const{ return "";}

private :
    std::string who_caught_it;
};

struct eoFitnessException : public eoException 
{
    eoFitnessException() : eoException()  {}
    eoFitnessException(const eoObject& caller) : eoException(caller) {}
};

struct eoNegativeFitnessException : public eoFitnessException
{
    eoNegativeFitnessException() : eoFitnessException()  {}
    eoNegativeFitnessException(const eoObject& caller) : eoFitnessException(caller) {}

    std::string what(void) const { return "negative fitness encountered"; }
};

struct eoMinimizingFitnessException : public eoFitnessException
{
    eoMinimizingFitnessException() : eoFitnessException()  {}
    eoMinimizingFitnessException(const eoObject& caller) : eoFitnessException(caller) {}
    
    std::string what(void) const { return "smaller fitness is better fitness, which is quite inappropriate here"; }
};

#endif
