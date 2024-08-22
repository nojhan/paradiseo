
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
        Johann Dreo <johann@dreo.fr>
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
        virtual std::shared_ptr<Itf> instantiate_ptr(bool no_cache = true) = 0;
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
        Itf& instantiate(bool no_cache = true) override
        {
            if(no_cache or not _instantiated) {
                // if(_instantiated) {
                    // delete _instantiated;
                // }
                _instantiated = op_constructor(_args);
                // _instantiated = op_constructor(_args);
            }
            return *_instantiated;
        }

        std::shared_ptr<Itf> instantiate_ptr(bool no_cache = true) override
        {
            if(no_cache or not _instantiated) {
                if(_instantiated) {
                    // delete _instantiated;
                }
                _instantiated = op_constructor(_args);
                // _instantiated = op_constructor(_args);
            }
            return _instantiated;
        }

        virtual ~eoForgeOperator() override
        {
            // delete _instantiated;
        }

    protected:
        std::tuple<Args...> _args;

    private:
        /** Metaprogramming machinery which deals with arguments lists @{ */
        template<class T>
        std::shared_ptr<Op> op_constructor(T& args)
        // Op* op_constructor(T& args)
        {
            // FIXME double-check that the copy-constructor is a good idea to make_from_tuple with dynamic storage duration.
            return std::make_shared<Op>(std::make_from_tuple<Op>(args));
            // return new Op(std::make_from_tuple<Op>(args));
        }
        /** @} */

    protected:
        std::shared_ptr<Itf> _instantiated;
        // Itf* _instantiated;
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

        Itf& instantiate( bool no_cache = true ) override
        {
            if(no_cache or not _instantiated) {
                // if(_instantiated) {
                    // delete _instantiated;
                // }
                _instantiated = std::shared_ptr<Op>();
                // _instantiated = new Op;
            }
            return *_instantiated;
        }

        std::shared_ptr<Itf> instantiate_ptr( bool no_cache = true ) override
        {
            if(no_cache or not _instantiated) {
                // if(_instantiated) {
                    // delete _instantiated;
                // }
                _instantiated = std::shared_ptr<Op>();
                // _instantiated = new Op;
            }
            return _instantiated;
        }

        virtual ~eoForgeOperator() override
        {
            // delete _instantiated;
        }

    protected:
        std::shared_ptr<Itf> _instantiated;
        // Itf* _instantiated;
};

/** A vector holding an operator (with deferred instantiation) at a given index.
 *
 * @note You can actually store several instances of the same class,
 * with different parametrization (or not).
 *
 * @warning When passing a reference (as it is often the case within ParadisEO),
 *          it is MANDATORY to wrap it in `std::ref`, or else it will default to use copy.
 *          This is is a source of bug which your compiler will fail to detect and that would
 *          disable any link between operators.
 *
 * @warning You may want to enable instantiation cache to grab some performances.
 *          The default is set to disable the cache, because its use with operators
 *          which hold a state will lead to unwanted behaviour.
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
template<class Itf, typename Enable = void>
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

         /** instantiate the operator managed at the given index.
          */
         Itf& instantiate(double index)
         {
             double frac_part, int_part;
             frac_part = std::modf(index, &int_part);
             if(frac_part != 0) {
                eo::log << eo::errors << "there is a fractional part in the given index (" << index << ")" << std::endl;
                assert(frac_part != 0);
             }
             return this->at(static_cast<size_t>(index))->instantiate(_no_cache);
         }

         std::shared_ptr<Itf> instantiate_ptr(double index)
         {
             double frac_part, int_part;
             frac_part = std::modf(index, &int_part);
             if(frac_part != 0) {
                eo::log << eo::errors << "there is a fractional part in the given index (" << index << ")" << std::endl;
                assert(frac_part != 0);
             }
             return this->at(static_cast<size_t>(index))->instantiate_ptr(_no_cache);
         }

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

        virtual ~eoForgeVector()
        {
            for(auto p : *this) {
                delete p;
            }
        }

    protected:
        bool _no_cache;
};

/** A map holding an operator (with deferred instantiation) at a given name.
 *
 * @note You can actually store several instances of the same class,
 * with different parametrization (or not).
 *
 * @warning When passing a reference (as it is often the case within ParadisEO),
 *          it is MANDATORY to wrap it in `std::ref`, or else it will default to use copy.
 *          This is is a source of bug which your compiler will fail to detect and that would
 *          disable any link between operators.
 *
 * @warning You may want to enable instantiation cache to grab some performances.
 *          The default is set to disable the cache, because its use with operators
 *          which hold a state will lead to unwanted behaviour.
 *
 * @code
    eoForgeMap<eoSelect<EOT>> factories(false);

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
template<class Itf, typename Enable = void>
class eoForgeMap : public std::map<std::string,eoForgeInterface<Itf>*>
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
        eoForgeMap( bool always_reinstantiate = true ) :
            _no_cache(always_reinstantiate)
        { }

         /** instantiate the operator managed at the given name.
          */
         Itf& instantiate(const std::string& name)
         {
             return this->at(name)->instantiate(_no_cache);
         }

        /** Add an operator to the list.
         *
         * @warning When passing a reference (as it is often the case within ParadisEO),
         *          it is MANDATORY to wrap it in `std::ref`, or else it will default to use copy.
         *          This is is a source of bug which your compiler will to detect and that would
         *          disable any link between operators.
         *
         */
        template<class Op, typename... Args>
        void add(const std::string& name, Args... args)
        {
            // We decay all args to ensure storing everything by value within the forge.
            // The references should thus be wrapped in a std::ref.
            auto pfo = new eoForgeOperator<Itf,Op,std::decay_t<Args>...>(
                    std::forward<Args>(args)...);
            this->insert({name, pfo});
        }

        /** Specialization for operators with empty constructors.
         */
        template<class Op>
        void add(const std::string& name)
        {
            eoForgeInterface<Itf>* pfo = new eoForgeOperator<Itf,Op>;
            this->insert({name, pfo});
        }

        /** Change the set up arguments to the constructor.
         *
         * @warning When passing a reference (as it is often the case within ParadisEO),
         *          it is MANDATORY to wrap it in `std::ref`, or else it will default to use copy.
         *          This is is a source of bug which your compiler will to detect and that would
         *          disable any link between operators.
         *
         * @warning The operator at `name` should have been added with eoForgeMap::add already..
         */
        template<class Op, typename... Args>
        void setup(const std::string& name, Args... args)
        {
            delete this->at(name); // Silent on nullptr.
            auto pfo = new eoForgeOperator<Itf,Op,std::decay_t<Args>...>(
                    std::forward<Args>(args)...);
            this->emplace({name, pfo});
        }

        /** Specialization for empty constructors.
         */
        template<class Op>
        void setup(const std::string& name)
        {
            delete this->at(name);
            auto pfo = new eoForgeOperator<Itf,Op>;
            this->emplace({name, pfo});
        }

        virtual ~eoForgeMap()
        {
            for(auto kv : *this) {
                delete kv.second;
            }
        }

    protected:
        bool _no_cache;
};


/** A range holding a parameter value at a given index.
 * 
 * This is essential a scalar numerical parameter, with bounds check
 * and an interface similar to an eoForgeVector.
 *
 * @note Contrary to eoForgeVector, this does not store a set of possible values.
 *
 * @code
    eoForgeScalar<double> factories(0.0, 1.0);

    // Actually instantiate.
    double param  = factories.instantiate(0.5);
 * @endcode
 *
 * @ingroup Foundry
 */
template<class Itf>
class eoForgeScalar
{
    public:
        using Interface = Itf;

        /** Constructor
         *
         * @param min Minimum possible value.
         * @param may Maximum possible value.
         */
        eoForgeScalar(Itf min, Itf max) :
            _min(min),
            _max(max)
        { }
        
        /** Just return the same value, without managing any instantiation.
         * 
         * Actually checks if value is in range.
         */
        Itf& instantiate(double value)
        {
            this->_value = value;
            if(not (_min <= value and value <= _max) ) {
                eo::log << eo::errors << "ERROR: the given value is out of range, I'll cap it." << std::endl;
                assert(_min <= value and value <= _max);
                if(value < _min) {
                    this->_value = _min;
                    return this->_value;
                }
                if(value > _max) {
                    this->_value = _max;
                    return this->_value;
                }
            }
            return this->_value;
        }

        Itf min() const { return _min; }
        Itf max() const { return _max; }

        /** Set the minimum possible value.
         */
        void min(Itf min)
        {
            assert(_min <= _max);
            _min = min;
        }
        
        /** Set the maximum possible value.
         */
        void max(Itf max)
        {
            assert(_max >= _min);
            _max = max;
        }

        /** Set the possible range of values.
         */
        void setup(Itf min, Itf max)
        {
            _min = min;
            _max = max;
            assert(_min <= _max);
        }

        // Nothing else, as it would not make sense.
    protected:
        Itf _value;

        Itf _min;
        Itf _max;
};

#endif // _eoForge_H_

