/***************************************************************************
 *   Copyright (C) 2008 by Waldo Cancino   *
 *   wcancino@icmc.usp.br   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef MOEOPTRCOMPARATOR_H_
#define MOEOPTRCOMPARATOR_H_

#include "moeoComparator.h"
/**
 * Functor allowing to compare two solutions.referenced by pointers.
 * Several MOEO related stuff have to sort populations according some criterion
 * Instead to do this, we used a vector whose elements are pointers to true individuals
 */

template < class MOEOT >
class moeoPtrComparator : public eoBF < const MOEOT *, const MOEOT *, bool >
{
public:

    /**
     * Ctor with a comparator
     * @param _cmp comparator to be employed
    */
    moeoPtrComparator( moeoComparator<MOEOT> & _cmp) : cmp(_cmp) {}

    /** compare two const individuals */
    bool operator() (const MOEOT *ptr1, const MOEOT *ptr2)
    {
        return cmp(*ptr1, *ptr2);
    }

    /** compare two non const individuals */
    bool operator() (MOEOT *ptr1, MOEOT *ptr2)
    {
        return cmp(*ptr1, *ptr2);
    }

private:
    moeoComparator<MOEOT> &cmp;
};

#endif /*MOEOPTRCOMPARATOR_H_*/
