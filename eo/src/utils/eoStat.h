// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoStat.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2000
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

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoStat_h
#define _eoStat_h

#include <eoFunctor.h>
#include <utils/eoParam.h>
#include <eoPop.h>

template <class EOT>
class eoStatBase : public eoUnaryFunctor<void, const eoPop<EOT>&>
{
public :
    virtual ~eoStatBase(){}

    /**
        calculate some statistic on the population
    */
    virtual void operator()(const eoPop<EOT>& _pop) = 0;
};

template <class EOT, class T>
class eoStat : public eoValueParam<T>, public eoStatBase<EOT>
{
public :
    eoStat(T _value, std::string _description) : eoValueParam<T>(_value, _description) {}
};

#include <numeric>

/**
    Average fitness of a population, fitness needs to be scalar.
*/
template <class EOT>
class eoAverageStat : public eoStat<EOT, double>
{
public :
    eoAverageStat(std::string _description = "Average Fitness") : eoStat<EOT, double>(0.0, _description) {}

    static double sumFitness(double _sum, const EOT& _eot)
    {
        _sum += _eot.fitness();
        return _sum;
    }

    eoAverageStat(double _value, std::string _desc) : eoStat<EOT, double>(_value, _desc) {}

    virtual void operator()(const eoPop<EOT>& _pop)
    {
        double v = std::accumulate(_pop.begin(), _pop.end(), 0.0, eoAverageStat::sumFitness);

        value() = v / _pop.size();
    }
};

template <class EOT>
class eoSecondMomentStats : public eoStat<EOT, std::pair<double, double> >
{
public :
    typedef std::pair<double, double> SquarePair;
    eoSecondMomentStats(std::string _description = "Average & Stdev") : eoStat<EOT, SquarePair>(std::make_pair(0.0,0.0), _description) {}

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
};

template <class EOT>
class eoNthElementFitnessStat : public eoStat<EOT, typename EOT::Fitness >
{
public :
    typedef typename EOT::Fitness Fitness;

    eoNthElementFitnessStat(int _which, std::string _description = "nth element fitness") : which(_which), eoStat<EOT, Fitness>(Fitness(), _description) {}

    virtual void operator()(const eoPop<EOT>& _pop)
    {
        if (which > _pop.size())
            throw logic_error("fitness requested of element outside of pop");

        value() = _pop.nth_element_fitness(which);
    }

private :
    unsigned which;
};


template <class EOT>
class eoBestFitnessStat : public eoNthElementFitnessStat<EOT>
{
public :
    typedef typename EOT::Fitness Fitness;

    eoBestFitnessStat(std::string _description = "Best Fitness") : eoNthElementFitnessStat<EOT>(0, _description) {}
};

template <class EOT>
class eoDistanceStat : public eoStat<EOT, double>
{
public :
    eoDistanceStat(std::string _name = "distance") : eoStat<EOT, double>(0.0, _name) {}

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
#endif
