// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoStat.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2000
// (c) 2010 Thales group
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

    Authors:
        todos@geneura.ugr.es, http://geneura.ugr.es
        Marc.Schoenauer@polytechnique.fr
        mkeijzer@dhi.dk
        Johann Dréo <johann.dreo@thalesgroup.com>
 */
//-----------------------------------------------------------------------------

#ifndef _eoStat_h
#define _eoStat_h

#include <numeric> // accumulate

#include <eoFunctor.h>
#include <utils/eoParam.h>
#include <eoPop.h>
#include <utils/eoMonitor.h>
//#include <utils/eoCheckPoint.h>

/** @defgroup Stats Statistics computation
 *
 * Compute various statistics on a population.
 *
 * Objects of those classes are generally called by an eoCheckPoint
 * to compute statistics about the population at a given generation.
 * As they inherit from eoValueParam, they can be printed drectly,
 * for instance by an eoMonitor.
 *
 * @see eoCheckPoint
 * @see eoMonitor
 *
 * @ingroup Utilities
 * @{
 */

/**
  Base class for all statistics that need to be calculated
  over the (unsorted) population
  (I guess it is not really necessary? MS.
  Depstd::ends, there might be reasons to have a stat that is not an eoValueParam,
  but maybe I'm just kidding myself, MK)
*/
template <class EOT>
class eoStatBase : public eoUF<const eoPop<EOT>&, void>
{
public:
  virtual void lastCall(const eoPop<EOT>&) {}
  virtual std::string className(void) const { return "eoStatBase"; }
};


template <class EOT> class eoCheckPoint;

/**
  The actual class that will be used as base for all statistics
  that need to be calculated over the (unsorted) population
  It is an eoStatBase AND an eoValueParam so it can be used in Monitors.
*/
template <class EOT, class T>
class eoStat : public eoValueParam<T>, public eoStatBase<EOT>
{
public:

    eoStat(T _value, std::string _description)
        : eoValueParam<T>(_value, _description)
        {}

    virtual std::string className(void) const
        { return "eoStat"; }


    eoStat<EOT, T>& addTo(eoCheckPoint<EOT>& cp)        { cp.add(*this);  return *this; }
    eoStat<EOT, T>& addTo(eoMonitor& mon)               { mon.add(*this); return *this; }
};



/**
  Base class for statistics calculated over a sorted snapshot of the population
*/
template <class EOT>
class eoSortedStatBase : public eoUF<const std::vector<const EOT*>&, void>
{
public:
  virtual void lastCall(const std::vector<const EOT*>&) {}
  virtual std::string className(void) const { return "eoSortedStatBase"; }

};

/**
  The actual class that will be used as base for all statistics
  that need to be calculated over the sorted population
  It's an eoSortedStatBase AND an eoValueParam so it can be used in Monitors.
*/
template <class EOT, class ParamType>
class eoSortedStat : public eoSortedStatBase<EOT>, public eoValueParam<ParamType>
{
public :
  eoSortedStat(ParamType _value, std::string _desc) : eoValueParam<ParamType>(_value, _desc) {}
  virtual std::string className(void) const { return "eoSortedStat"; }

    eoSortedStat<EOT, ParamType>& addTo(eoCheckPoint<EOT>& cp)        { cp.add(*this);  return *this; }
    eoSortedStat<EOT, ParamType>& addTo(eoMonitor& mon)               { mon.add(*this); return *this; }
};

/**
   Average fitness of a population. Fitness can be:
   - double
   - eoMinimizingFitness or eoMaximizingFitness
     The average of each objective is evaluated.

   ( For eoScalarFitnessAssembled user eoAssembledFitnessStat classes.)
*/
#if  defined(_MSC_VER) && (_MSC_VER < 1300)
template <class EOT> class eoAverageStat : public eoStat<EOT, EOT::Fitness>
#else
template <class EOT> class eoAverageStat : public eoStat<EOT, typename EOT::Fitness>
#endif
{
public :

    using eoStat<EOT, typename EOT::Fitness>::value;

    typedef typename EOT::Fitness Fitness;

    eoAverageStat(std::string _description = "Average Fitness")
      : eoStat<EOT, Fitness>(Fitness(), _description) {}

    static Fitness sumFitness(double _sum, const EOT& _eot){
        _sum += _eot.fitness();
        return _sum;
    }

    eoAverageStat(double _value, std::string _desc) : eoStat<EOT, double>(_value, _desc) {}

    virtual void operator()(const eoPop<EOT>& _pop){
      doit(_pop, Fitness()); // specializations for scalar and std::vector
    }

  virtual std::string className(void) const { return "eoAverageStat"; }

private :

    // Default behavior
    template <class T>
    void doit(const eoPop<EOT>& _pop, T)
    {
        Fitness v = std::accumulate(_pop.begin(), _pop.end(), Fitness(0.0), eoAverageStat::sumFitness);

        value() = v / _pop.size();
    }

};

/**
    Average fitness + Std. dev. of a population, fitness needs to be scalar.
*/
template <class EOT>
class eoSecondMomentStats : public eoStat<EOT, std::pair<double, double> >
{
public :

    using eoStat<EOT, std::pair<double, double> >::value;

    typedef typename EOT::Fitness fitness_type;

    typedef std::pair<double, double> SquarePair;

    eoSecondMomentStats(std::string _description = "Average & Stdev")
        : eoStat<EOT, SquarePair>(std::make_pair(0.0,0.0), _description)
        {}

    static SquarePair sumOfSquares(SquarePair _sq, const EOT& _eo)
    {
        double fitness = _eo.fitness();

        _sq.first += fitness;
        _sq.second += fitness * fitness;
        return _sq;
    }

    virtual void operator()(const eoPop<EOT>& _pop)
    {
        SquarePair result = std::accumulate(_pop.begin(), _pop.end(), std::make_pair(0.0, 0.0), eoSecondMomentStats::sumOfSquares);

        double n = _pop.size();
        value().first = result.first / n; // average
        value().second = sqrt( (result.second - n * value().first * value().first) / (n - 1.0)); // stdev
    }

  virtual std::string className(void) const { return "eoSecondMomentStats"; }
};

/**
    The n_th element fitness in the population (see eoBestFitnessStat)
*/
#if  defined(_MSC_VER) && (_MSC_VER < 1300)
template <class EOT>
class eoNthElementFitnessStat : public eoSortedStat<EOT, EOT::Fitness >
#else
template <class EOT>
class eoNthElementFitnessStat : public eoSortedStat<EOT, typename EOT::Fitness >
#endif
{
public :
    using eoSortedStat<EOT, typename EOT::Fitness >::value;

    typedef typename EOT::Fitness Fitness;

    eoNthElementFitnessStat(unsigned _whichElement, std::string _description = "nth element fitness")
      : eoSortedStat<EOT, Fitness>(Fitness(), _description), whichElement(_whichElement) {}

    virtual void operator()(const std::vector<const EOT*>& _pop)
    {
        if (whichElement > _pop.size())
            throw std::logic_error("fitness requested of element outside of pop");

        doit(_pop, Fitness());
    }

  virtual std::string className(void) const { return "eoNthElementFitnessStat"; }
private :

    struct CmpFitness
    {
      CmpFitness(unsigned _whichElement, bool _maxim) : whichElement(_whichElement), maxim(_maxim) {}

      bool operator()(const EOT* a, const EOT* b)
      {
        if (maxim)
          return a->fitness()[whichElement] > b->fitness()[whichElement];

        return a->fitness()[whichElement] < b->fitness()[whichElement];
      }

      unsigned whichElement;
      bool maxim;
    };

    // for everything else
    template <class T>
    void doit(const std::vector<const EOT*>& _pop, T)
    {
      value() = _pop[whichElement]->fitness();
    }

    unsigned whichElement;
};

/* Actually, you shouldn't need to sort the population to get the best fitness
   MS - 17/11/00

   But then again, if another stat needs sorted fitness anyway, getting the best
   out would be very fast.
   MK - 09/01/03

template <class EOT>
class eoBestFitnessStat : public eoStat<EOT, typename EOT::Fitness >
{
public :
    typedef typename EOT::Fitness Fitness;

    eoBestFitnessStat(std::string _description = "Best Fitness") :
      eoStat<EOT, Fitness>(Fitness(), _description) {}

    virtual void operator()(const eoPop<EOT>& _pop)
    {
        value() = _pop.nth_element_fitness(0);
    }

};
*/

/**
   Best fitness of a population. Fitness can be:
   - double
   - eoMinimizingFitness or eoMaximizingFitness

   ( For eoScalarFitnessAssembled look at eoAssembledFitnessStat )
*/

#if  defined(_MSC_VER) && (_MSC_VER < 1300)
template <class EOT>
class eoBestFitnessStat : public eoStat<EOT, EOT::Fitness>
#else
template <class EOT>
class eoBestFitnessStat : public eoStat<EOT, typename EOT::Fitness>
#endif
{
public:

    using eoStat<EOT, typename EOT::Fitness>::value;

    typedef typename EOT::Fitness Fitness;

    eoBestFitnessStat(std::string _description = "Best ")
        : eoStat<EOT, Fitness>(Fitness(), _description)
        {}

    void operator()(const eoPop<EOT>& _pop) {
        doit(_pop, Fitness() ); // specializations for scalar and std::vector
    }

    virtual std::string className(void) const { return "eoBestFitnessStat"; }


private :

    struct CmpFitness
    {
      CmpFitness(unsigned _which, bool _maxim) : which(_which), maxim(_maxim) {}

      bool operator()(const EOT& a, const EOT& b)
      {
        if (maxim)
          return a.fitness()[which] < b.fitness()[which];

        return a.fitness()[which] > b.fitness()[which];
      }

      unsigned which;
      bool maxim;
    };

    // default
    template<class T>
    void doit(const eoPop<EOT>& _pop, T)
    { // find the largest elements
      value() = _pop.best_element().fitness();
    }

};
/** @example t-eoSSGA.cpp
 */

template <class EOT>
class eoDistanceStat : public eoStat<EOT, double>
{
public:

    using eoStat<EOT, double>::value;

    eoDistanceStat(std::string _name = "distance")
        : eoStat<EOT, double>(0.0, _name)
        {}

    template <class T>
    double distance(T a, T b)
    {
        T res = a-b;
        return res < 0? -res : res;
    }

    double distance(bool a, bool b)
    {
        return (a==b)? 0 : 1;
    }

    void operator()(const eoPop<EOT>& _pop)
    {
        double& v = value();
        v = 0.0;

        for (unsigned i = 0; i < _pop.size(); ++i)
        {
            for (unsigned j = 0; j < _pop.size(); ++j)
            {
                for (unsigned k = 0; k < _pop[i].size(); ++k)
                {
                    v += distance(_pop[i][k], _pop[j][k]);
                }
            }
        }

        double sz = _pop.size();
        v /= sz * sz * _pop[0].size();
    }
  virtual std::string className(void) const { return "eoDistanceStat"; }

};



/*
template <class EOT>
class eoStdevStat : public eoStat<EOT, double >
{
public :
    typedef typename eoSecondMomentStats<EOT>::SquarePair SquarePair;

    eoStdevStat(std::string _description = "Stdev") : eoStat<EOT, double>(0.0, _description) {}

    virtual void operator()(const eoPop<EOT>& _pop)
    {
        SquarePair result = std::accumulate(pop.begin(), pop.end(), std::make_pair(0.0, 0.0), eoSecondMomentStats::sumOfSquares);

        double n = pop.size();
        value() = sqrt( (result.second - (result.first / n)) / (n - 1.0)); // stdev
    }
};
*/


//! A robust measure of dispersion (also called midspread or middle fifty) that is the difference between the third and the first quartile.
template<class EOT>
class eoInterquartileRangeStat : public eoStat< EOT, typename EOT::Fitness >
{
public:
    using eoStat<EOT, typename EOT::Fitness>::value;

    eoInterquartileRangeStat( typename EOT::Fitness start, std::string description = "IQR" ) : eoStat<EOT,typename EOT::Fitness>( start, description ) {}

    virtual void operator()( const eoPop<EOT> & _pop )
    {
        if( _pop.size() == 0 ) {
            // how to implement value() = 0 ?

        } else {
            eoPop<EOT> pop = _pop;

            unsigned int quartile = pop.size()/4;
            std::nth_element( pop.begin(), pop.begin()+quartile*1, pop.end() );
            typename EOT::Fitness Q1 = pop[quartile].fitness();

            std::nth_element( pop.begin(), pop.begin()+quartile*3, pop.end() );
            typename EOT::Fitness Q3 = pop[quartile*3].fitness();

            value() = Q3 - Q1;
        }
    }

    virtual std::string className(void) const { return "eoInterquartileRangeStat"; }
};
/** @example t-eoIQRStat.cpp
 */

/** Compute the average size of indivudals over the population
 *
 * Obviously, will work only on representations that implement the (standard) "size()" method,
 * like any STL container.
 */
template<class EOT>
class eoAverageSizeStat : public eoStat< EOT, double>
{
public:

    using eoStat<EOT, double>::value;

    eoAverageSizeStat( std::string description = "Av.Size" ) :
        eoStat<EOT,double>( 0.0, description ) {} // 0 by default

    virtual void operator()( const eoPop<EOT> & pop )
    {
        size_t pop_size = pop.size();

        std::vector<size_t> sizes;
        sizes.reserve(pop_size);

        for( unsigned int i=0, s = pop_size; i<s; ++i ) {
            sizes.push_back( pop[i].size() );
        }

        size_t sum = std::accumulate( sizes.begin(), sizes.end(), 0 );

        value() = static_cast<double>(sum) / static_cast<double>(pop_size);
    }

    virtual std::string className(void) const { return "eoAverageSizeStat"; }
};

/** @} */
#endif
