
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

#include <memory>
#include <map>
#include <any>
#include <string>
#include <tuple>

/** Interface for a "Forge": a class that can defer instanciation of EO's operator.
 *
 * This interface only declares an `instanciate` method,
 * in order to be able to make containers of factories (@see eoForgeOperator).
 *
 * @ingroup Core
 * @defgroup Forge Wrap and defer operators' instanciations.
 * @ingroup Forge
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
   //                ^ desired     ^ to-be-instanciated         ^ operator's
   //                  interface     operator                     parameters

   // Actual instanciation:
   eoSelect<EOT>& select = forge.instanciate();
 * @endcode
 *
 * @ingroup Forge
 */
template<class Itf, class Op, typename... Args>
class eoForgeOperator : public eoForgeInterface<Itf>
{
    public:
        eoForgeOperator(Args&&... args) :
            _args(std::forward<Args>(args)...)
        { }

        Itf& instanciate()
        {
            return *(constructor(_args));
        }

        virtual ~eoForgeOperator() {}

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
            return p_op;
        }

        template <typename... Ts>
        Op* constructor(std::tuple<Ts...>& args)
        {
            return constructor(args, gen_seq<sizeof...(Ts)>{});
        }

};

/** A map holding an operator with deferred instanciation at a given key.
 *
 * @note You can actually store several instances of the same class,
 * with different parametrization (or not).
 *
 * @code
    eoForgeMap<eoSelect<EOT>> named_factories;

    // Capture constructor's parameters and defer instanciation.
    named_factories.add<eoRankMuSelect<EOT>>("RMS", 1);
    named_factories.setup<eoRankMuSelect<EOT>>("RMS", 5); // Edit

    // Actually instanciate.
    eoSelect<EOT>& op  = named_factories.instanciate("RMS");

    // Call.
    op();
 * @endcode
 *
 * @ingroup Forge
 */
template<class Itf>
class eoForgeMap : public std::map<std::string, std::shared_ptr<eoForgeInterface<Itf>> >
{
    public:
        template<class Op, typename... Args>
        void setup(std::string key, Args&&... args)
        {
            auto opf = std::make_shared< eoForgeOperator<Itf,Op,Args...> >(std::forward<Args>(args)...);
            (*this)[key] = opf;
        }

        template<class Op, typename... Args>
        void add(std::string key, Args&&... args)
        {
            setup<Op>(key, std::forward<Args>(args)...);
        }

        Itf& instanciate(std::string key)
        {
            return this->at(key)->instanciate();
        }
};

/** A vector holding an operator with deferred instanciation at a given index.
 *
 * @note You can actually store several instances of the same class,
 * with different parametrization (or not).
 *
 * @code
    eoForgeVector<eoSelect<EOT>> named_factories;

    // Capture constructor's parameters and defer instanciation.
    named_factories.add<eoRankMuSelect<EOT>>(1);
    named_factories.setup<eoRankMuSelect<EOT>>(0, 5); // Edit

    // Actually instanciate.
    eoSelect<EOT>& op  = named_factories.instanciate("RMS");

    // Call.
    op();
 * @endcode
 *
 * @ingroup Forge
 */
template<class Itf>
class eoForgeVector : public std::vector<std::shared_ptr<eoForgeInterface<Itf>> >
{
    public:
        template<class Op, typename... Args>
        void add(Args&&... args)
        {
            auto opf = std::make_shared< eoForgeOperator<Itf,Op,Args...> >(std::forward<Args>(args)...);
            this->push_back(opf);
        }

        template<class Op, typename... Args>
        void setup(size_t index, Args&&... args)
        {
            auto opf = std::make_shared< eoForgeOperator<Itf,Op,Args...> >(std::forward<Args>(args)...);
            this->at(index) = opf;
        }

        Itf& instanciate(size_t index)
        {
            return this->at(index)->instanciate();
        }
};

#endif // _eoForge_H_

