/*

(c) 2010 Thales group

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; version 2
    of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>

*/

#ifndef _eoDualFitness_h_
#define _eoDualFitness_h_

#include <functional>
#include <iostream>
#include <utility> // for std::pair
#include <string>

#include <utils/eoStat.h>
#include <utils/eoLogger.h>

/** @addtogroup Evaluation
 * @{
 */

//! A fitness class that permits to compare feasible and unfeasible individuals and guaranties that a feasible individual will always be better than an unfeasible one.
/**
 * Use this class as fitness if you have some kind of individuals
 * that must be always considered as better than others while having the same fitness type.
 *
 * Wraps a scalar fitness _values such as a double or int, with the option of
 * maximizing (using less<BaseType>, @see eoMaximizingDualFitness)
 * or minimizing (using greater<BaseType>, @see eoMinimizingDualFitness).
 *
 * Suitable constructors, assignments and casts are defined to work
 * with those quantities as if they were a pair of: a BaseType and a boolean.
 *
 * When changing the fitness, you can use:
 *     individual.fitness( std::make_pair<BaseType,bool>( fitness, feasibility ) );
 *
 * Be aware that, when printing or reading an eDualFitness instance on a iostream,
 * friend IO classes use a space separator.
 *
 * This class overrides operator<() to use the Compare template argument and handle feasibility.
 * Over operators are coded using this sole function.
 *
 * Standard arithmetic operators are provided to add or substract dual fitnesses.
 * They behave as expected for the fitness value and gives priority to unfeasible fitness
 * (i.e. when adding or substracting dual fitness, the only case when the result will be
 *  a feasible fitness is when both are feasible, else the result is an unfeasibe fitness)
*/
template <class BaseType, class Compare >
class eoDualFitness
{
protected:
    //! Scalar type of the fitness (generally a double)
    BaseType _value;

    //! Flag that marks if the individual is feasible
    bool _is_feasible;

    /** Flag to prevent partial initialization
     *
     * The reason behind the use of this flag is a bit complicated.
     * Normally, we would not want to allow initialization on a scalar.
     * But in MOEO, this would necessitate to re-implement most of the
     * operator computing metrics, as they expect generic scalars.
     *
     * As this would be too much work, we use derived metric classes and
     * overload them so that they initialize dual fitnesses with the
     * feasibility flag. But the compiler still must compile the base
     * methods, that use the scalar interface.
     *
     * Thus, eoDualFitness has a scalar interface, but this flag add a
     * security against partial initialization. In DEBUG mode, asserts
     * will fail if the feasibility has not been explicitly initialized
     * at runtime.
     */
    bool _feasible_init;

public:

    //! Empty initialization
    /*!
     * Unfeasible by default
     */
    eoDualFitness() :
        _value(0.0),
        _is_feasible(false),
        _feasible_init(false)
    {}

    //! Initialization with only the value, the fitness will be unfeasible.
    /*!
     * WARNING: this is what is used when you initialize a new fitness from a double.
     * If you use this interface, you MUST set the feasibility BEFORE
     * asking for it or the value. Or else, an assert will fail in debug mode.
     */
    template<class T>
    eoDualFitness( T value ) :
        _value(value),
        _is_feasible(false),
        _feasible_init(false)
    {
    }


    //! Copy constructor
    eoDualFitness(const eoDualFitness& other) :
        _value(other._value),
        _is_feasible(other._is_feasible),
        _feasible_init(true)
    {}

    //! Constructor from explicit value/feasibility
    eoDualFitness(const BaseType& v, const bool& is_feasible) :
        _value(v),
        _is_feasible(is_feasible),
        _feasible_init(true)
    {}

    //! From a std::pair (first element is the value, second is the feasibility)
    eoDualFitness(const std::pair<BaseType,bool>& dual) :
        _value(dual.first),
        _is_feasible(dual.second),
        _feasible_init(true)
    {}

    /** Conversion operator: it permits to use a fitness instance as  its  scalar
     * type, if needed. For example, this is possible:
     *     eoDualFitness<double,std::less<double> > fit;
     *     double val = 1.0;
     *     val = fit;
     */
    operator BaseType(void) const { return _value; }


    inline bool is_feasible() const
    {
        assert( _feasible_init );
        return _is_feasible;
    }

    //! Explicitly set the feasibility. Useful if you have used previously the instantiation on a single scalar.
    inline void is_feasible( bool feasible )
    {
        this->is_feasible( feasible );
        this->_feasible_init = true;
    }

    inline BaseType value() const
    {
        assert( _feasible_init );
        return _value;
    }

    //! Copy operator from a std::pair
    eoDualFitness& operator=( const std::pair<BaseType, bool>& v )
    {
        this->_value = v.first;
        this->is_feasible( v.second );
        return *this;
    }

    //! Copy operator from another eoDualFitness
    template <class F, class Cmp>
    eoDualFitness<BaseType,Compare> & operator=( const eoDualFitness<BaseType, Compare>& other )
    {
        if (this != &other) {
            this->_value = other._value;
            this->is_feasible( other.is_feasible() );
        }
        return *this;
    }

    //! Copy operator from a scalar
    template<class T>
    eoDualFitness& operator=(const T v)
    {
        this->_value = v;
        this->_is_feasible = false;
        this->_feasible_init = false;
        return *this;
    }

    //! Comparison that separate feasible individuals from unfeasible ones. Feasible are always better
    /*!
     * Use less as a default comparison operator
     * (see the "Compare" template of the class to change this behaviour,
     * @see eoMinimizingDualFitness for an example).
     */
    bool operator<(const eoDualFitness& other) const
    {
        // am I better (less, by default) than the other ?

        // if I'm feasible and the other is not
        if( this->is_feasible() && !other.is_feasible() ) {
            // no, the other has a better fitness
            return false;

        } else if( !this->is_feasible() && other.is_feasible() ) {
            // yes, a feasible fitness is always better than an unfeasible one
            return true;

        } else {
            // the two fitness are of the same type
            // lets rely on the comparator
            return Compare()(_value, other._value);
        }
    }

    //! Greater: if the other is lesser than me
    bool operator>( const eoDualFitness& other ) const  { return other < *this; }

    //! Less or equal: if the other is not lesser than me
    bool operator<=( const eoDualFitness& other ) const { return !(other < *this); }

    //! Greater or equal: if the other is not greater than me
    bool operator>=(const eoDualFitness& other ) const { return !(*this < other); }

    //! Equal: if the other is equal to me
    bool operator==(const eoDualFitness& other) const { return ( _is_feasible == other._is_feasible ) && ( _value == other._value ); }

public:

    /* FIXME it would be better to raise errors (or warnings) if one try to apply arithmetics operators between feasible
     * and unfeasible fitnesses. This necessitates to add wrappers for operators that aggregates sets of dual fitnesses
     * (like eoStat), both for separating feasibility and for aggregating them.
     */

    // NOTE: we cannot declare this set of operator classes as friend, because there is two differerent templated classes declared later
    // (for minimizing and maximizing)

    //! Add a given fitness to the current one
    template<class T>
    eoDualFitness<BaseType,Compare> & operator+=( const T that )
    {
        this->_value += that;
        return *this;
    }

    //! Add a given fitness to the current one
    eoDualFitness<BaseType,Compare> & operator+=( const eoDualFitness<BaseType,Compare> & that )
    {
        // from._value += that._value;
        this->_value += that._value;

        // true only if the two are feasible, else false
        // from._is_feasible = from._is_feasible && that._is_feasible;
        this->_is_feasible = this->_is_feasible && that._is_feasible;

        return *this;
    }

    //! Substract a given fitness to the current one
    template<class T>
    eoDualFitness<BaseType,Compare> & operator-=( const T that )
    {
        this->_value -= that;
        return *this;
    }

    //! Substract a given fitness to the current one
    eoDualFitness<BaseType,Compare> & operator-=( const eoDualFitness<BaseType,Compare> & that )
    {
        this->_value -= that._value;

        // true only if the two are feasible, else false
        this->_is_feasible = this->_is_feasible && that._is_feasible;

        return *this;
    }


    //! Add a given fitness to the current one
    template<class T>
    eoDualFitness<BaseType,Compare> & operator/=( T that )
    {
        this->_value /= that;
        return *this;
    }

    //! Add a given fitness to the current one
    eoDualFitness<BaseType,Compare> & operator/=( const eoDualFitness<BaseType,Compare> & that )
    {
        this->_value /= that._value;

        // true only if the two are feasible, else false
        this->_is_feasible = this->_is_feasible && that._is_feasible;

        return *this;
    }

    template<class T>
    eoDualFitness<BaseType,Compare> operator+( T that )
    {
        this->_value += that;
        return *this;
    }

    // Add this fitness's value to that other, and return a _new_ instance with the result.
    eoDualFitness<BaseType,Compare> operator+( const eoDualFitness<BaseType,Compare> & that )
    {
        eoDualFitness<BaseType,Compare> from( *this );
        return from += that;
    }

    template<class T>
    eoDualFitness<BaseType,Compare> operator-( T that )
    {
        this->_value -= that;
        return *this;
    }

    // Add this fitness's value to that other, and return a _new_ instance with the result.
    eoDualFitness<BaseType,Compare> operator-( const eoDualFitness<BaseType,Compare> & that )
    {
        eoDualFitness<BaseType,Compare> from( *this );
        return from -= that;
    }


    template<class T>
    eoDualFitness<BaseType,Compare> operator/( T that )
    {
        this->_value /= that;
        return *this;
    }

    // Add this fitness's value to that other, and return a _new_ instance with the result.
    eoDualFitness<BaseType,Compare> operator/( const eoDualFitness<BaseType,Compare> & that )
    {
        eoDualFitness<BaseType,Compare> from( *this );
        return from /= that;
    }

    //! Print an eoDualFitness instance as a pair of numbers, separated by a space
    friend
    std::ostream& operator<<( std::ostream& os, const eoDualFitness<BaseType,Compare> & fitness )
    {
        os << fitness._value << " " << fitness.is_feasible();
        return os;
    }

    //! Read an eoDualFitness instance as a pair of numbers, separated by a space
    friend
    std::istream& operator>>( std::istream& is, eoDualFitness<BaseType,Compare> & fitness )
    {
        BaseType value;
        is >> value;

        bool feasible;
        is >> feasible;

        fitness._value = value;
        fitness.is_feasible( feasible );
        return is;
    }
};

//! Compare dual fitnesses as if we were maximizing
typedef eoDualFitness<double, std::less<double> >    eoMaximizingDualFitness;

//! Compare dual fitnesses as if we were minimizing
typedef eoDualFitness<double, std::greater<double> > eoMinimizingDualFitness;

//! A predicate that returns the feasibility of a given dual fitness
/** Use this in STL algorithm that use binary predicates (e.g. count_if, find_if, etc.)
 */
template< class EOT>
bool eoIsFeasible ( const EOT & sol ) { return sol.fitness().is_feasible(); }


/** Separate the population into two: one with only feasible individuals, the other with unfeasible ones.
 */
template<class EOT>
class eoDualPopSplit : public eoUF<const eoPop<EOT>&, void>
{
protected:
    eoPop<EOT> _pop_feasible;
    eoPop<EOT> _pop_unfeasible;

public:
    //! Split the pop and keep them in members
    void operator()( const eoPop<EOT>& pop )
    {
        _pop_feasible.clear();
        _pop_feasible.reserve(pop.size());

        _pop_unfeasible.clear();
        _pop_unfeasible.reserve(pop.size());

        for( typename eoPop<EOT>::const_iterator ieot=pop.begin(), iend=pop.end(); ieot!=iend; ++ieot ) {
            /*
            if( ieot->invalid() ) {
                eo::log << eo::errors << "ERROR: trying to access to an invalid fitness" << std::endl;
            }
            */
            if( ieot->fitness().is_feasible() ) {
                _pop_feasible.push_back( *ieot );
            } else {
                _pop_unfeasible.push_back( *ieot );
            }
        }
    }

    //! Merge feasible and unfeasible populations into a new one
    eoPop<EOT> merge() const
    {
        eoPop<EOT> merged;
        merged.reserve( _pop_feasible.size() + _pop_unfeasible.size() );
        std::copy(   _pop_feasible.begin(),   _pop_feasible.end(), std::back_inserter<eoPop<EOT> >(merged) );
        std::copy( _pop_unfeasible.begin(), _pop_unfeasible.end(), std::back_inserter<eoPop<EOT> >(merged) );
        return merged;
    }

    eoPop<EOT>&   feasible() { return   _pop_feasible; }
    eoPop<EOT>& unfeasible() { return _pop_unfeasible; }
};


/** Embed two eoStat and call the first one on the feasible individuals and
 * the second one on the unfeasible ones, merge the two resulting value in
 * a string, separated by a given marker.
 */
template<class EOT, class EOSTAT>
class eoDualStatSwitch : public eoStat< EOT, std::string >
{
protected:
    EOSTAT & _stat_feasible;
    EOSTAT & _stat_unfeasible;

    std::string _sep;

    eoDualPopSplit<EOT> _pop_split;

public:
    using eoStat<EOT,std::string>::value;

    eoDualStatSwitch( EOSTAT & stat_feasible,  EOSTAT & stat_unfeasible, std::string sep=" "  ) :
        eoStat<EOT,std::string>(
                "?"+sep+"?",
                stat_feasible.longName()+sep+stat_unfeasible.longName()
                                ),
        _stat_feasible(stat_feasible),
        _stat_unfeasible(stat_unfeasible),
        _sep(sep)
    { }

    virtual void operator()( const eoPop<EOT> & pop )
    {
        // create two separated pop in this operator
        _pop_split( pop );

          _stat_feasible( _pop_split.feasible() );
        _stat_unfeasible( _pop_split.unfeasible() );

        std::ostringstream out;
        out << _stat_feasible.value() << _sep << _stat_unfeasible.value();

        value() = out.str();
    }
};

/** @} */
#endif // _eoDualFitness_h_
