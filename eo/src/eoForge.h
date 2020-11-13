
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

#ifndef _eoForge_H_
#define _eoForge_H_

#include <string>
#include <tuple>
#include <utility>

// In case you want to debug arguments captured in tuples:
// template<typename Type, unsigned N, unsigned Last>
// struct tuple_printer {
//
//     static void print(std::ostream& out, const Type& value) {
//         out << std::get<N>(value) << ", ";
//         tuple_printer<Type, N + 1, Last>::print(out, value);
//     }
// };
//
// template<typename Type, unsigned N>
// struct tuple_printer<Type, N, N> {
//
//     static void print(std::ostream& out, const Type& value) {
//         out << std::get<N>(value);
//     }
//
// };
//
// template<typename... Types>
// std::ostream& operator<<(std::ostream& out, const std::tuple<Types...>& value) {
//     out << "(";
//     tuple_printer<std::tuple<Types...>, 0, sizeof...(Types) - 1>::print(out, value);
//     out << ")";
//     return out;
// }

/**
 * @defgroup Foundry
 *
 * Tools for automatic algorithms assembling, selection and search.
 */

/** Interface for a "Forge": a class that can defer instantiation of EO's operator.
 *
 * This interface only declares an `instantiate` method,
 * in order to be able to make containers of factories (@see eoForgeOperator).
 *
 * @ingroup Core
 * @ingroup Foundry
 */
template<class Itf>
class eoForgeInterface
{
    public:
        virtual Itf& instantiate(bool no_cache = true) = 0;
        virtual ~eoForgeInterface() {}
};

/** This "Forge" can defer the instantiation of an EO's operator.
 *
 * It allows to decouple the constructor's parameters setup from its actual call.
 * You can declare a parametrized operator at a given time,
 * then actually instantiate it (with the given interface) at another time.
 *
 * This allows for creating containers of pre-parametrized operators (@see eoForgeVector).
 *
 * @warning When passing a reference (as it is often the case within ParadisEO),
 * it is MANDATORY to wrap it in `std::ref`, or else it will default to use copy.
 * This is is a source of bug which your compiler will to detect and that would
 * disable any link between operators.
 *
 * @code
   eoForgeOperator<eoselect<EOT>,eoRankMuSelect<EOT>> forge(mu);
   //                ^ desired     ^ to-be-instantiated     ^ operator's
   //                  interface     operator                 parameters
   // Actual instantiation:
   eoSelect<EOT>& select = forge.instantiate();
 * @endcode
 *
 * @warning You may want to enable instantiation cache to grab some performances.
 * The default is set to disable the cache, because its use with operators
 * which hold a state will lead to unwanted behaviour.
 *
 * @ingroup Foundry
 */
template<class Itf, class Op, typename... Args>
class eoForgeOperator : public eoForgeInterface<Itf>
{
    public:
        // Use an additional template to avoid redundant copies of decayed Args variadic.
        template<class ...Args2>
        eoForgeOperator(Args2... args) :
            _args(std::forward<Args2>(args)...),
            _instantiated(nullptr)
        { }

        /** instantiate the managed operator class.
         *
         * That is call its constructor with the set up arguments.
         *
         * @warning Do not enable cache with operators which hold a state.
         *
         * @param no_cache If false, will enable caching previous instances.
         */
        Itf& instantiate(bool no_cache = true)
        {
            if(no_cache or not _instantiated) {
                if(_instantiated) {
                    delete _instantiated;
                }
                _instantiated = op_constructor(_args);
            }
            return *_instantiated;
        }

        virtual ~eoForgeOperator()
        {
            delete _instantiated;
        }

    protected:
        std::tuple<Args...> _args;

    private:
        /** Metaprogramming machinery which deals with arguments lists @{ */
        template<class T>
        Op* op_constructor(T& args)
        {
            // FIXME double-check that the copy-constructor is a good idea to make_from_tuple with dynamic storage duration.
            return new Op(std::make_from_tuple<Op>(args));
        }
        /** @} */

    protected:
        Itf* _instantiated;
};

/** Partial specialization for constructors without any argument.
 */
template<class Itf, class Op>
class eoForgeOperator<Itf,Op> : public eoForgeInterface<Itf>
{
    public:
        eoForgeOperator() :
            _instantiated(nullptr)
        { }

        Itf& instantiate( bool no_cache = true )
        {
            if(no_cache or not _instantiated) {
                if(_instantiated) {
                    delete _instantiated;
                }
                _instantiated = new Op;
            }
            return *_instantiated;
        }

        virtual ~eoForgeOperator()
        {
            delete _instantiated;
        }

    protected:
        Itf* _instantiated;
};

/** A vector holding an operator with deferred instantiation at a given index.
 *
 * @note You can actually store several instances of the same class,
 * with different parametrization (or not).
 *
 * @warning When passing a reference (as it is often the case within ParadisEO),
 * it is MANDATORY to wrap it in `std::ref`, or else it will default to use copy.
 * This is is a source of bug which your compiler will to detect and that would
 * disable any link between operators.
 *
 * @warning You may want to enable instantiation cache to grab some performances.
 * The default is set to disable the cache, because its use with operators
 * which hold a state will lead to unwanted behaviour.
 *
 * @code
    eoForgeVector<eoSelect<EOT>> factories(false);

    // Capture constructor's parameters and defer instantiation.
    factories.add<eoRankMuSelect<EOT>>(1);
    factories.setup<eoRankMuSelect<EOT>>(0, 5); // Edit

    // Actually instantiate.
    eoSelect<EOT>& op  = factories.instantiate(0);

    // Call.
    op();
 * @endcode
 *
 * @ingroup Foundry
 */
template<class Itf>
class eoForgeVector : public std::vector<eoForgeInterface<Itf>*>
{
    public:
        using Interface = Itf;
        /** Default constructor do not cache instantiations.
         *
         * @warning
         * You most probably want to disable caching for operators that hold a state.
         * If you enable the cache, the last used instantiation will be used,
         * at its last state.
         * For example, continuators should most probably not be cached,
         * as they very often hold a state in the form of a counter.
         * At the end of a search, the continuator will be in the end state,
         * and thus always ask for a stop.
         * Reusing an instance in this state will de facto disable further searches.
         *
         * @param always_reinstantiate If false, will enable cache for the forges in this container.
         */
        eoForgeVector( bool always_reinstantiate = true ) :
            _no_cache(always_reinstantiate)
        { }

        /** Add an operator to the list.
         *
         * @warning When passing a reference (as it is often the case within ParadisEO),
         * it is MANDATORY to wrap it in `std::ref`, or else it will default to use copy.
         * This is is a source of bug which your compiler will to detect and that would
         * disable any link between operators.
         *
         */
        template<class Op, typename... Args>
        void add(Args... args)
        {
            // We decay all args to ensure storing everything by value within the forge.
            // The references should thus be wrapped in a std::ref.
            auto pfo = new eoForgeOperator<Itf,Op,std::decay_t<Args>...>(
                    std::forward<Args>(args)...);
            this->push_back(pfo);
        }

        /** Specialization for operators with empty constructors.
         */
        template<class Op>
        void add()
        {
            eoForgeInterface<Itf>* pfo = new eoForgeOperator<Itf,Op>;
            this->push_back(pfo);
        }

        /** Change the set up arguments to the constructor.
         *
         * @warning When passing a reference (as it is often the case within ParadisEO),
         * it is MANDATORY to wrap it in `std::ref`, or else it will default to use copy.
         * This is is a source of bug which your compiler will to detect and that would
         * disable any link between operators.
         *
         * @warning The operator at `index` should have been added with eoForgeVector::add already..
         */
        template<class Op, typename... Args>
        void setup(size_t index, Args... args)
        {
            assert(index < this->size());
            delete this->at(index); // Silent on nullptr.
            auto pfo = new eoForgeOperator<Itf,Op,std::decay_t<Args>...>(
                    std::forward<Args>(args)...);
            this->at(index) = pfo;
        }

        /** Specialization for empty constructors.
         */
        template<class Op>
        void setup(size_t index)
        {
            assert(index < this->size());
            delete this->at(index);
            auto pfo = new eoForgeOperator<Itf,Op>;
            this->at(index) = pfo;
        }

        /** instantiate the operator managed at the given index.
         */
        Itf& instantiate(size_t index)
        {
            return this->at(index)->instantiate(_no_cache);
        }

        virtual ~eoForgeVector()
        {
            for(auto p : *this) {
                delete p;
            }
        }

    protected:
        bool _no_cache;
};

#endif // _eoForge_H_

