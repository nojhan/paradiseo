
/*
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

   © 2020 Thales group

    Authors:
        Johann Dreo <johann.dreo@thalesgroup.com>
*/

#ifndef _eoAlgoFoundry_H_
#define _eoAlgoFoundry_H_

#include <vector>

/** A vector of eoForge which hold an index.
 *
 * To be used in conjunction with a subclass of an eoAlgoFoundry,
 * where it can store all the alternative operators
 * and hold the link to the encoding. @see eoAlgoFoundryEA
 *
 * As with eoForgeVector, adding a managed operator
 * is done through public member variable's `add` method,
 * which takes the class name as template and its constructor's parameters
 * as arguments. For example:
 * @code
 * eoOperatorFoundry< eoSelectOne<EOT> > selectors;
 * selectors.add< eoStochTournamentSelect<EOT> >( 0.5 );
 * @endcode
 *
 * @warning If the managed constructor takes a reference YOU SHOULD ABSOLUTELY wrap it
 * in a `std::ref` when using `add` or `setup`, or it will silently be passed as a copy,
 * which would effectively disable any link between operators.
 *
 * @ingroup Core
 * @ingroup Foundry
 */
template<class Itf>
class eoOperatorFoundry : public eoForgeVector< Itf >
{
    public:
        eoOperatorFoundry(size_t encoding_index, bool always_reinstantiate = true ) :
            eoForgeVector<Itf>(always_reinstantiate),
            _index(encoding_index)
        { }

        size_t index() const { return _index; }

    protected:
        //! Unique index in the eoAlgoFoundry.
        size_t _index;
};

template<class Itf>
class eoParameterFoundry : public eoForgeScalar< Itf >
{
    static_assert(std::is_arithmetic<Itf>::value,
        "eoParameterFoundry should only be used on arithmetic types (i.e. integer or floating point types)");

    public:
        eoParameterFoundry(size_t encoding_index, Itf min, Itf max) :
            eoForgeScalar<Itf>(min, max),
            _index(encoding_index)
        { }

        size_t index() const { return _index; }

    protected:
        //! Unique index in the eoAlgoFoundry.
        size_t _index;
};

/** Interface of a Foundry: a class that instantiate an eoAlgo on-the-fly, given a choice of its operators.
 *
 * The chosen operators are encoded in a vector of indices.
 *
 * The foundry subclass should first be set up with sets of operators of the same interface,
 * held within an eoOperatorFoundry member.
 * @code
 * eoOperatorFoundry< eoSelectOne<EOT> > selectors;
 * @endcode
 *
 * In a second step, the operators to be used should be selected
 * by indicating their index, just like if the foundry was an array:
 * @code
 * foundry.select({0, 1, 2});
 * //              ^  ^  ^
 * //              |  |  |
 * //              |  |  + 3d operator
 * //              |  + 2d operator
 * //              + 1st operator
 * @endcode
 *
 * If you don't (want to) recall the order of the operators in the encoding,
 * you can use the `index()` member of eoOperatorFoundry, for example:
 * @code
 * foundry.at(foundry.continuators.index()) = 2; // select the third continuator
 * @endcode
 *
 * Now, you must implement the foundry just like any eoAlgo, by using the eoPop interface:
 * @code
 * foundry(pop);
 * @encode
 * It will instantiate the needed operators (only) and the algorithm itself on-the-fly,
 * and then run it.
 *
 * @note: Thanks to the underlying eoOperatorFoundry, not all the added operators are instantiated.
 * Every instantiation is deferred upon actual use. That way, you can still reconfigure them
 * at any time with `eoForgeOperator::setup`, for example:
 * @code
 * foundry.selector.at(0).setup(0.5); // using constructor's arguments
 * @endcode
 *
 * @warning If the managed constructor takes a reference YOU SHOULD ABSOLUTELY wrap it
 * in a `std::ref` when using `add` or `setup`, or it will silently be passed as a copy,
 * which would effectively disable any link between operators.
 *
 * @ingroup Core
 * @ingroup Foundry
 * @ingroup Algorithms
 */
template<class EOT>
class eoAlgoFoundry : public eoAlgo<EOT>
{
    public:
        /** 
         */
        eoAlgoFoundry( size_t nb_operators ) :
            _size(nb_operators),
            _encoding(_size,0)
        { }

        /** Select indices of all the operators.
         *
         * i.e. Select an algorithm to instantiate.
         */
        void select( std::vector<size_t> encoding )
        {
            assert(encoding.size() == _encoding.size());
            _encoding = encoding;
        }

        /** Access to the index of the currently selected operator.
         */
        size_t& at(size_t i)
        {
            return _encoding.at(i);
        }

        size_t size() const
        {
            return _size;
        }

        /** Return the underlying encoding vector.
         */
        std::vector<size_t> encoding() const
        {
            return _encoding;
        }

    protected:
        const size_t _size;
        std::vector<size_t> _encoding;

};

#endif // _eoAlgoFoundry_H_
