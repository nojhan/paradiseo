// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/*
(c) Thales group, 2013

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
Lionel Parreaux <lionel.parreaux@gmail.com>

*/

/** @defgroup Logging Logging
 * @ingroup Utilities



@{
*/

#ifndef _EOOPTIONAL_H
#define _EOOPTIONAL_H

#include <eoObject.h>


template< class T >
class eoOptional {
public:
	static const eoOptional<T> null; // = eoOptional<T>();
	
	eoOptional (T& init)
	: _val(&init)
	{ }
	
	bool hasValue() const
	{
		return _val != NULL;
	}
	
	T& get () const
	{
		if (!hasValue())
			throw std::runtime_error("Cannot get a reference from a eoOptional wrapper with no value");
		return *_val;
	}
	
protected:
	eoOptional ()
	: _val(NULL)
	{ }
	
private:
	T* _val;
};

template< class T >
const eoOptional<T> eoOptional<T>::null = eoOptional<T>();


#endif // _EOOPTIONAL_H


/** @} */

