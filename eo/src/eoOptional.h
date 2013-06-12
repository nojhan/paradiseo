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

/**

A utility class for wrapping non-const references and use them as default arguments in functions.

For example, this is not valid C++98 code:

\code
struct MyClass {
    MyClass(T& my_T = default_T)
    : actual_T(my_T)
    { }
private:
    T default_T;
    T& actual_T;
};
\endcode

This is the same code using eoOptional, which is valid:

\code
struct MyClass {
    MyClass(eoOptional<T> my_T = NULL)
    : actual_T(my_T.getOr(default_T))
    { }
private:
    T default_T;
    T& actual_T;
};
\endcode

And from the point of view of the user, it is transparent:

\code
// Three ways of using MyClass:
MyClass mc1;
MyClass mc2(NULL);
T t;
MyClass mc3(t);
\endcode


@ingroup Utilities
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

    // used mainly for converting NULL to this class
    eoOptional (T* init)
    : _val(init)
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

    T& getOr (T& default) const
    {
        return hasValue()? *_val: default;
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

