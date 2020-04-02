
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
 * @defgroup Foundry Tools for automatic algorithms assembling, selection and search.
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
        virtual Itf& instanciate() = 0;
        virtual ~eoForgeInterface() {}
};

/** This "Forge" can defer the instanciation of an EO's operator.
 *
 * It allows to decouple the constructor's parameters setup from its actual call.
 * You can declare a parametrized operator at a given time,
 * then actually instanciate it (with the given interface) at another time.
 *
 * This allows for creating containers of pre-parametrized operators (@see eoForgeMap or @see eoForgeVector).
 *
 * @code
   eoForgeOperator<eoselect<EOT>,eoRankMuSelect<EOT>> forge(mu);
   //                ^ desired     ^ to-be-instanciated     ^ operator's
   //                  interface     operator                 parameters
   // Actual instanciation:
   eoSelect<EOT>& select = forge.instanciate();
 * @endcode
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

        Itf& instanciate()
        {
            if(not _instanciated) {
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

    protected:
        Itf* _instanciated;
};

//! Partial specialization for constructors without any argument.
template<class Itf, class Op>
class eoForgeOperator<Itf,Op> : public eoForgeInterface<Itf>
{
    public:
        eoForgeOperator() :
            _instanciated(nullptr)
        { }

        Itf& instanciate()
        {
            if(not _instanciated) {
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
 * @code
    eoForgeVector<eoSelect<EOT>> factories;

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
        template<class Op, typename... Args>
        void add(Args... args)
        {
            auto pfo = new eoForgeOperator<Itf,Op,Args...>(args...);
            this->push_back(pfo);
        }

        template<class Op>
        void add()
        {
            eoForgeInterface<Itf>* pfo = new eoForgeOperator<Itf,Op>;
            this->push_back(pfo);
        }

        template<class Op, typename... Args>
        void setup(size_t index, Args... args)
        {
            assert(this->at(index) != nullptr);
            delete this->at(index);
            auto pfo = new eoForgeOperator<Itf,Op,Args...>(args...);
            this->at(index) = pfo;
        }

        Itf& instanciate(size_t index)
        {
            return this->at(index)->instanciate();
        }

        virtual ~eoForgeVector()
        {
            for(auto p : *this) {
                delete p;
            }
        }

};

#endif // _eoForge_H_

