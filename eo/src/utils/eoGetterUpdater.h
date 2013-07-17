/*

(c) Thales group, 2010

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

#ifndef _eoGetterUpdater_h
#define _eoGetterUpdater_h

#include <utils/eoUpdater.h>

template <class EOT> class eoCheckPoint;

/**
    eoGetterUpdater is an eoUpdater that updates its eoValueParam::value() from
    the specified object and the specified getter method on this object.
    
    @ingroup Utilities
*/
template <class T, class V = double>
class eoGetterUpdater : public eoUpdater, public eoValueParam<V>
{
public:
    using eoValueParam<V>::value;
    
    virtual std::string className(void) const { return "eoGetterUpdater"; }
    
    typedef V (T::*MethodType)();
    
    eoGetterUpdater(T& _instance, MethodType _method)
    : instance(_instance), method(_method)
    { }
    
    // Overload to accept const getter methods; safely casts them to non-const
    eoGetterUpdater(T& _instance, V (T::*_method)() const)
    : instance(_instance), method((MethodType)_method)
    { }
    
    virtual void operator()()
    {
        value() = (instance.*method)();
    }
    
private:
    T& instance;
    MethodType method; // ie:  V (T::*method)();
};



#endif















