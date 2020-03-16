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

Copyright (C) 2020 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef _edoCombinedContinue_h
#define _edoCombinedContinue_h

#include <eoFunctor.h>
#include <eoPersistent.h>

/** Combine several EDO continuators in a single one.
 *
 * Return true if any of the managed continuator ask for a stop.
 *
 * @see edoContinue
 *
 * @ingroup Continuators
 * @ingroup Core
 */
template<class D>
class edoCombinedContinue : public edoContinue<D>, public std::vector<edoContinue<D>*>
{
public:
    edoCombinedContinue( edoContinue<D>& cont ) :
        edoContinue<D>(),
        std::vector<edoContinue<D>*>(1,&cont)
    { }

    edoCombinedContinue( std::vector<edoContinue<D>*> conts ) :
        edoContinue<D>(),
        std::vector<edoContinue<D>*>(conts)
    { }

    void add( edoContinue<D>& cont)
    {
        this->push_back(&cont);
    }

    bool operator()(const D& distrib)
    {
        for(const auto cont : *this) {
            if( not (*cont)(distrib)) {
                return false;
            }
        }
        return true;
    }

    virtual std::string className() const { return "edoCombinedContinue"; }
};

#endif // !_edoCombinedContinue_h
