
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

/** Interface for a "Forge": a class that can defer instanciation of EO's operator.
 *
 * This interface only declares an `instanciate` method,
 * in order to be able to make containers of factories (@see eoForgeOperator).
 *
 * @ingroup Core
 * @ingroup Foundry
 */
template<class Itf>
class eoForgeInterface
{
    public:
        virtual Itf& instanciate(bool no_cache = true) = 0;
        virtual ~eoForgeInterface() {}
};

/** This "Forge" can defer the instanciation of an EO's operator.
 *
 * It allows to decouple the constructor's parameters setup from its actual call.
 * You can declare a parametrized operator at a given time,
 * then actually instanciate it (with the given interface) at another time.
 *
 * This allows for creating containers of pre-parametrized operators (@see eoForgeVector).
 *
 * @code
   eoForgeOperator<eoselect<EOT>,eoRankMuSelect<EOT>> forge(mu);
   //                ^ desired     ^ to-be-instanciated     ^ operator's
   //                  interface     operator                 parameters
   // Actual instanciation:
   eoSelect<EOT>& select = forge.instanciate();
 * @endcode
 *
 * @warning You may want to enable instanciation cache to grab some performances.
 * The default is set to disable the cache, because its use with operators
 * which hold a state will lead to unwanted behaviour.
 *
 * @ingroup Foundry
 */
template<class Itf, class Op, typename... Args>
class eoForgeOperator : public eoForgeInterface<Itf>
{
    public:
        eoForgeOperator(Args... args) :
            _args(args...),
            _instanciated(nullptr)
        { }

        /** Instanciate the managed operator class.
         *
         * That is call its constructor with the set up arguments.
         *
         * @warning Do not enable cache with operators which hold a state.
         *
         * @param no_cache If false, will enable caching previous instances.
         */
        Itf& instanciate(bool no_cache = true)
        {
            if(no_cache or not _instanciated) {
                if(_instanciated) {
                    delete _instanciated;
                }
                _instanciated = constructor(_args);
            }
            return *_instanciated;
        }

        virtual ~eoForgeOperator()
        {
            delete _instanciated;
        }

    protected:
        std::tuple<Args...> _args;

    private:
        /** Metaprogramming machinery which deals with arguments lists @{ */
        template <int... Idx>
        struct index {};

        template <int N, int... Idx>
        struct gen_seq : gen_seq<N - 1, N - 1, Idx...> {};

        template <int... Idx>
        struct gen_seq<0, Idx...> : index<Idx...> {};

        template <typename... Ts, int... Idx>
        Op* constructor(std::tuple<Ts...>& args, index<Idx...>)
        {
            Op* p_op = new Op(std::get<Idx>(args)...);
            _instanciated = p_op;
            return p_op;
        }

        template <typename... Ts>
        Op* constructor(std::tuple<Ts...>& args)
        {
            return constructor(args, gen_seq<sizeof...(Ts)>{});
        }
        /** @} */

    protected:
        Itf* _instanciated;
};

/** Partial specialization for constructors without any argument.
 */
template<class Itf, class Op>
class eoForgeOperator<Itf,Op> : public eoForgeInterface<Itf>
{
    public:
        eoForgeOperator() :
            _instanciated(nullptr)
        { }

        Itf& instanciate( bool no_cache = true )
        {
            if(no_cache or not _instanciated) {
                if(_instanciated) {
                    delete _instanciated;
                }
                _instanciated = new Op;
            }
            return *_instanciated;
        }

        virtual ~eoForgeOperator()
        {
            delete _instanciated;
        }

    protected:
        Itf* _instanciated;
};

/** A vector holding an operator with deferred instanciation at a given index.
 *
 * @note You can actually store several instances of the same class,
 * with different parametrization (or not).
 *
 * @warning You may want to enable instanciation cache to grab some performances.
 * The default is set to disable the cache, because its use with operators
 * which hold a state will lead to unwanted behaviour.
 *
 * @code
    eoForgeVector<eoSelect<EOT>> factories(false);

    // Capture constructor's parameters and defer instanciation.
    factories.add<eoRankMuSelect<EOT>>(1);
    factories.setup<eoRankMuSelect<EOT>>(0, 5); // Edit

    // Actually instanciate.
    eoSelect<EOT>& op  = factories.instanciate(0);

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
        /** Default constructor do not cache instanciations.
         *
         * @warning
         * You most probably want to disable caching for operators that hold a state.
         * If you enable the cache, the last used instanciation will be used,
         * at its last state.
         * For example, continuators should most probably not be cached,
         * as they very often hold a state in the form of a counter.
         * At the end of a search, the continuator will be in the end state,
         * and thus always ask for a stop.
         * Reusing an instance in this state will de facto disable further searches.
         *
         * @param always_reinstanciate If false, will enable cache for the forges in this container.
         */
        eoForgeVector( bool always_reinstanciate = true ) :
            _no_cache(always_reinstanciate)
        { }

        /** Add an operator to the list.
         */
        template<class Op, typename... Args>
        void add(Args... args)
        {
            auto pfo = new eoForgeOperator<Itf,Op,Args...>(args...);
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
         * @warning The operator at `index` should have been added with eoForgeVector::add already..
         */
        template<class Op, typename... Args>
        void setup(size_t index, Args... args)
        {
            assert(this->at(index) != nullptr);
            delete this->at(index);
            auto pfo = new eoForgeOperator<Itf,Op,Args...>(args...);
            this->at(index) = pfo;
        }

        /** Instanciate the operator managed at the given index.
         */
        Itf& instanciate(size_t index)
        {
            return this->at(index)->instanciate(_no_cache);
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

