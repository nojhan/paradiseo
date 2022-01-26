
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
   © 2022 Institut Pasteur

    Authors:
        Johann Dreo <johann.dreo@thalesgroup.com>
*/

#ifndef _eoAlgoFoundry_H_
#define _eoAlgoFoundry_H_

#include <vector>
#include <variant>

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
 *   eoOperatorFoundry< eoSelectOne<EOT> > selectors;
 *   selectors.add< eoStochTournamentSelect<EOT> >( 0.5 );
 * @endcode
 *
 * @warning If the managed constructor takes a reference YOU SHOULD ABSOLUTELY wrap it
 *          in a `std::ref` when using `add` or `setup`, or it will silently be passed as a copy,
 *          which would effectively disable any link between operators.
 *
 * @ingroup Core
 * @ingroup Foundry
 */
template<class Itf>
class eoOperatorFoundry : public eoForgeVector< Itf >
{
    public:
        /** Constructor
         * 
         * @param encoding_index The slot position in the encodings, at which this operator is held.
         * @param always_reinstantiate If false, will enable cache for the forges in this container.
         */
        eoOperatorFoundry(size_t encoding_index, bool always_reinstantiate = true ) :
            eoForgeVector<Itf>(always_reinstantiate),
            _index(encoding_index)
        { }

        /** Returns the slot index at which this is registered.
         */
        size_t index() const { return _index; }

    protected:
        //! Unique index in the eoAlgoFoundry.
        size_t _index;
};


/** A vector of eoForge which hold a scalar numeric value.
 *
 * To be used in conjunction with a subclass of an eoAlgoFoundry,
 * where it can hold a range of parameter values 
 * and hold the link to the encoding. @see eoAlgoFoundryEA
 *
 * As with eoForgeScalar, managed parameters
 * are represented through a [min,max] range.
 * 
 * For example:
 * @code
 *   eoParameterFoundry< double > proba(0.0, 1.0);
 * @endcode
 *
 * @ingroup Core
 * @ingroup Foundry
 */
template<class Itf>
class eoParameterFoundry : public eoForgeScalar< Itf >
{
    static_assert(std::is_arithmetic<Itf>::value,
        "eoParameterFoundry should only be used on arithmetic types (i.e. integer or floating point types)");

    public:
        /** Underlying type of the parameter.
         * 
         * @note: You probably only want to use either `double` or `size_t`.
         * @see eoAlgoFoundry
         */
        using Type = Itf;

        /** Constructor
         * 
         * @param encoding_index The slot position in the encodings, at which this parameter is held.
         * @param min Minimium possible value.
         * @param max Maximum possible value.
         */
        eoParameterFoundry(size_t encoding_index, Itf min, Itf max) :
            eoForgeScalar<Itf>(min, max),
            _index(encoding_index)
        { }

        /** Returns the slot index at which this is registered.
         */
        size_t index() const { return _index; }

    protected:
        //! Unique index in the eoAlgoFoundry.
        size_t _index;
};

/** Interface of a Foundry: a class that instantiate an eoAlgo on-the-fly, given a choice of its operators.
 *
 * The chosen operators are encoded in a vector of numbers.
 *
 * The foundry subclass should first be set up with sets of operators of the same interface,
 * held within an eoOperatorFoundry member.
 * @code
 *   eoOperatorFoundry< eoSelectOne<EOT> > selectors;
 * @endcode
 *
 * In a second step, the operators to be used should be selected
 * by indicating their index, just like if the foundry was an array:
 * @code
 *   foundry.select({size_t{0}, size_t{1}, size_t{2}});
 *   //                     ^          ^          ^
 *   //                     |          |          |
 *   //                     |          |          + 3d operator
 *   //                     |          + 2d operator
 *   //                     + 1st operator
 * @endcode
 *
 * If you don't (want to) recall the order of the operators in the encoding,
 * you can use the `index()` member of eoOperatorFoundry, for example:
 * @code
 *   foundry.at(foundry.continuators.index()) = size_t{2}; // select the third continuator
 * @endcode
 *
 * Now, you must implement the foundry just like any eoAlgo, by using the eoPop interface:
 * @code
 *   foundry(pop);
 * @encode
 * It will instantiate the needed operators (only) and the algorithm itself on-the-fly,
 * and then run it.
 * 
 * @note: The "encoding" which represent the selected options, figuring the actual meta-algorithm,
 *        is a vector of `std::variant`, which can hold either a `size_t` or a `double`.
 *        The first one is used to indicate the index of an operator class
 *        *or* a parameter which is a size.
 *        The second is used to store numerical parameters values.
 *
 * @note: Thanks to the underlying eoOperatorFoundry, not all the added operators are instantiated.
 *        Every instantiation is deferred upon actual use. That way, you can still reconfigure them
 *        at any time with `eoForgeOperator::setup`, for example:
 *        @code
 *        foundry.selector.at(0).setup(0.5); // using constructor's arguments
 *        @endcode
 *
 * @warning If the managed constructor takes a reference YOU SHOULD ABSOLUTELY wrap it
 *          in a `std::ref` when using `add` or `setup`, or it will silently be passed as a copy,
 *          which would effectively disable any link between operators.
 *
 * @ingroup Core
 * @ingroup Foundry
 * @ingroup Algorithms
 */
template<class EOT>
class eoAlgoFoundry : public eoAlgo<EOT>
{
    public:
        // We could use `std::any` instead of a variant,
        // but this would be more prone to errors from the end user, at the end.
        // Either the encoding is an index (of the operator within the list of instances)
        // either it's a real-valued parameter,
        // either it's a size.
        // So there's no need for more types (AFAIK).
        
        /** The type use to represent a selected option in the meta-algorithm.
         * 
         * This can figure, either:
         * - the index of an operator in the list of possible ones,
         * - the actual value of a numeric paramater,
         * - the value of a parameter which is a size.
         */
        using Encoding = std::variant<size_t, double>;

        /** The type use to store all selected options.
         */
        using Encodings = std::vector<Encoding>;
        
        /** Constructor.
         * 
         * @param nb_slots Number of operators or parameters that are assembled to make an algorithm.
         */
        eoAlgoFoundry( size_t nb_slots ) :
            _size(nb_slots)
        { }

        /** Select indices of all the operators.
         *
         * i.e. Select an algorithm to instantiate.
         * 
         * @note: You need to indicate the type of each item
         *        if you want to call this with a brace-initialized vector.
         * 
         * For example:
         * @code
         *   foundry.select({ size_t{1}, double{0.5}, size_t{3} });
         * @endcode
         * 
         * Or you can initialize the vector first:
         * @code
         *   double crossover_rate = 0.5;
         *   size_t crossover_oper = 3;
         *   eoAlgoFoundry<EOT>::Encodings encoded_algo(foundry.size());
         *   encoded_algo[foundry.crossover_rates.index()] = crossover_rate;
         *   encoded_algo[foundry.crossover_opers.index()] = crossover_oper;
         * @encdoe
         */
        void select( Encodings encodings )
        {
            assert(encodings.size() == _size);
            _encodings = encodings;
        }

        /** Access to the encoding of the currently selected operator.
         * 
         * @warning: This returns a `std::variant`, which you should `std::get<T>`.
         * 
         * For example:
         * @code
         *   size_t opera_id = std::get<size_t>(foundry.at(2));
         *   double param_id = std::get<double>(foundry.at(3));
         * @endcode
         * 
         * @note: You can use rank, value or len to have automatic casting.
         */
        Encoding & at(size_t i)
        {
            return _encodings.at(i);
        }

        /** Access to the currently selected ID of an operator.
         */
        template<class OP>
        size_t rank(const OP& op)
        {
            return std::get<size_t>( at(op.index()) );
        }

        /** Access to the currently selected value of a numeric parameter.
         */
        template<class OP>
        double value(const OP& param)
        {
            return std::get<double>( at(param.index()) );
        }

        /** Access to the currently selected value of a unsigned integer parameter.
         */
        template<class OP>
        size_t len(const OP& param)
        {
            return std::get<size_t>( at(param.index()) );
        }
        
        /** Returns the number of slots that makes this algorithm.
         */
        size_t size() const
        {
            return _size;
        }

        /** Return the underlying encoding vector.
         */
        Encodings encodings() const
        {
            return _encodings;
        }

    protected:
        const size_t _size;
        std::vector<Encoding> _encodings;

};

#endif // _eoAlgoFoundry_H_
