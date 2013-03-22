/*
(c) Benjamin Bouvier, 2013

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
    Benjamin Bouvier <benjamin.bouvier@gmail.com>
*/
# ifndef __EOSERIAL_TRAITS_H__
# define __EOSERIAL_TRAITS_H__

/**
 * @file Traits.h
 *
 * Traits used for serialization purposes.
 *
 * @author Benjamin Bouvier <benjamin.bouvier@gmail.com>
 */

namespace eoserial
{

    /**
     * @brief Trait to know whether Derived is derived from Base or not.
     *
     * To know whether A is derived from B, just test the boolean IsDerivedFrom<A, B>::value.
     *
     * @see http://www.gotw.ca/publications/mxc++-item-4.htm
     */
    template<class Derived, class Base>
    class IsDerivedFrom
    {
        struct no{};
        struct yes{ no _[2]; };

        static yes Test( Base* something );
        static no Test( ... );

    public:
        enum { value = sizeof( Test( static_cast<Derived*>(0) ) ) == sizeof(yes) };
    };

}

# endif // __EOSERIAL_TRAITS_H__
