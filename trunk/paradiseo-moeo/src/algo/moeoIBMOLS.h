// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoIBMOLS.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOIBMOLS_H_
#define MOEOIBMOLS_H_

#include <math.h>
#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoPop.h>
#include <moMove.h>
#include <moMoveInit.h>
#include <moNextMove.h>
#include <algo/moeoLS.h>
#include <archive/moeoArchive.h>
#include <fitness/moeoBinaryIndicatorBasedFitnessAssignment.h>
#include <move/moeoMoveIncrEval.h>

/**
 * Indicator-Based Multi-Objective Local Search (IBMOLS) as described in
 * Basseur M., Burke K. : "Indicator-Based Multi-Objective Local Search" (2007).
 */
template < class MOEOT, class Move >
class moeoIBMOLS : public moeoLS < MOEOT, eoPop < MOEOT > & >
{
public:

    /** The type of objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Ctor.
     * @param _moveInit the move initializer
     * @param _nextMove the neighborhood explorer
     * @param _eval the full evaluation
     * @param _moveIncrEval the incremental evaluation
     * @param _fitnessAssignment the fitness assignment strategy
     * @param _continuator the stopping criteria
     */
    moeoIBMOLS(
        moMoveInit < Move > & _moveInit,
        moNextMove < Move > & _nextMove,
        eoEvalFunc < MOEOT > & _eval,
        moeoMoveIncrEval < Move > & _moveIncrEval,
        moeoBinaryIndicatorBasedFitnessAssignment < MOEOT > & _fitnessAssignment,
        eoContinue < MOEOT > & _continuator
    ) :
            moveInit(_moveInit),
            nextMove(_nextMove),
            eval(_eval),
            moveIncrEval(_moveIncrEval),
            fitnessAssignment (_fitnessAssignment),
            continuator (_continuator)
    {}


    /**
     * Apply the local search until a local archive does not change or
     * another stopping criteria is met and update the archive _arch with new non-dominated solutions.
     * @param _pop the initial population
     * @param _arch the (updated) archive
     */
    void operator() (eoPop < MOEOT > & _pop, moeoArchive < MOEOT > & _arch)
    {
        // evaluation of the objective values
        /*
                for (unsigned int i=0; i<_pop.size(); i++)
                {
                    eval(_pop[i]);
                }
        */
        // fitness assignment for the whole population
        fitnessAssignment(_pop);
        // creation of a local archive
        moeoArchive < MOEOT > archive;
        // creation of another local archive (for the stopping criteria)
        moeoArchive < MOEOT > previousArchive;
        // update the archive with the initial population
        archive.update(_pop);
        unsigned int count = 0;
        do
        {
            previousArchive.update(archive);
            oneStep(_pop);
            count++;
            archive.update(_pop);
        } while ( (! archive.equals(previousArchive)) && (continuator(_arch)) );
        _arch.update(archive);
        cout << "\t" << count << " steps" << endl;
    }


private:

    /** the move initializer */
    moMoveInit < Move > & moveInit;
    /** the neighborhood explorer */
    moNextMove < Move > & nextMove;
    /** the full evaluation */
    eoEvalFunc < MOEOT > & eval;
    /** the incremental evaluation */
    moeoMoveIncrEval < Move > & moveIncrEval;
    /** the fitness assignment strategy */
    moeoBinaryIndicatorBasedFitnessAssignment < MOEOT > & fitnessAssignment;
    /** the stopping criteria */
    eoContinue < MOEOT > & continuator;


    /**
     * Apply one step of the local search to the population _pop
     * @param _pop the population
     */
    void old_oneStep (eoPop < MOEOT > & _pop)
    {
        // the move
        Move move;
        // the objective vector and the fitness of the current solution
        ObjectiveVector x_objVec;
        double x_fitness;
        // the index, the objective vector and the fitness of the worst solution in the population (-1 implies that the worst is the newly created one)
        int worst_idx;
        ObjectiveVector worst_objVec;
        double worst_fitness;
////////////////////////////////////////////
        // the indexes and the objective vectors of the extreme non-dominated points
        int ext_0_idx, ext_1_idx;
        ObjectiveVector ext_0_objVec, ext_1_objVec;
        unsigned int ind;
////////////////////////////////////////////
        // the index of the current solution to be explored
        unsigned int i=0;
        // initilization of the move for the first individual
        moveInit(move, _pop[i]);
        while (i<_pop.size() && continuator(_pop))
        {
            // x = one neigbour of pop[i]
            // evaluate x in the objective space
            x_objVec = moveIncrEval(move, _pop[i]);
            // update every fitness values to take x into account and compute the fitness of x
            x_fitness = fitnessAssignment.updateByAdding(_pop, x_objVec);

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // extreme solutions (min only!)
            ext_0_idx = -1;
            ext_0_objVec = x_objVec;
            ext_1_idx = -1;
            ext_1_objVec = x_objVec;
            for (unsigned int k=0; k<_pop.size(); k++)
            {
                // ext_0
                if (_pop[k].objectiveVector()[0] < ext_0_objVec[0])
                {
                    ext_0_idx = k;
                    ext_0_objVec = _pop[k].objectiveVector();
                }
                else if ( (_pop[k].objectiveVector()[0] == ext_0_objVec[0]) && (_pop[k].objectiveVector()[1] < ext_0_objVec[1]) )
                {
                    ext_0_idx = k;
                    ext_0_objVec = _pop[k].objectiveVector();
                }
                // ext_1
                else if (_pop[k].objectiveVector()[1] < ext_1_objVec[1])
                {
                    ext_1_idx = k;
                    ext_1_objVec = _pop[k].objectiveVector();
                }
                else if ( (_pop[k].objectiveVector()[1] == ext_1_objVec[1]) && (_pop[k].objectiveVector()[0] < ext_1_objVec[0]) )
                {
                    ext_1_idx = k;
                    ext_1_objVec = _pop[k].objectiveVector();
                }
            }
            // worst init
            if (ext_0_idx == -1)
            {
                ind = 0;
                while (ind == ext_1_idx)
                {
                    ind++;
                }
                worst_idx = ind;
                worst_objVec = _pop[ind].objectiveVector();
                worst_fitness = _pop[ind].fitness();
            }
            else if (ext_1_idx == -1)
            {
                ind = 0;
                while (ind == ext_0_idx)
                {
                    ind++;
                }
                worst_idx = ind;
                worst_objVec = _pop[ind].objectiveVector();
                worst_fitness = _pop[ind].fitness();
            }
            else
            {
                worst_idx = -1;
                worst_objVec = x_objVec;
                worst_fitness = x_fitness;
            }
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // who is the worst ?
            for (unsigned int j=0; j<_pop.size(); j++)
            {
                if ( (j!=ext_0_idx) && (j!=ext_1_idx) )
                {
                    if (_pop[j].fitness() < worst_fitness)
                    {
                        worst_idx = j;
                        worst_objVec = _pop[j].objectiveVector();
                        worst_fitness = _pop[j].fitness();
                    }
                }
            }
            // if the worst solution is the new one
            if (worst_idx == -1)
            {
                // if all its neighbours have been explored,
                // let's explore the neighborhoud of the next individual
                if (! nextMove(move, _pop[i]))
                {
                    i++;
                    if (i<_pop.size())
                    {
                        // initilization of the move for the next individual
                        moveInit(move, _pop[i]);
                    }
                }
            }
            // if the worst solution is located before _pop[i]
            else if (worst_idx <= i)
            {
                // the new solution takes place insteed of _pop[worst_idx]
                _pop[worst_idx] = _pop[i];
                move(_pop[worst_idx]);
                _pop[worst_idx].objectiveVector(x_objVec);
                _pop[worst_idx].fitness(x_fitness);
                // let's explore the neighborhoud of the next individual
                i++;
                if (i<_pop.size())
                {
                    // initilization of the move for the next individual
                    moveInit(move, _pop[i]);
                }
            }
            // if the worst solution is located after _pop[i]
            else if (worst_idx > i)
            {
                // the new solution takes place insteed of _pop[i+1] and _pop[worst_idx] is deleted
                _pop[worst_idx] = _pop[i+1];
                _pop[i+1] = _pop[i];
                move(_pop[i+1]);
                _pop[i+1].objectiveVector(x_objVec);
                _pop[i+1].fitness(x_fitness);
                // let's explore the neighborhoud of the individual _pop[i+2]
                i += 2;
                if (i<_pop.size())
                {
                    // initilization of the move for the next individual
                    moveInit(move, _pop[i]);
                }
            }
            // update fitness values
            fitnessAssignment.updateByDeleting(_pop, worst_objVec);
        }
    }













// INUTILE !!!!






    /**
     * Apply one step of the local search to the population _pop
     * @param _pop the population
     */
    void new_oneStep (eoPop < MOEOT > & _pop)
    {
        // the move
        Move move;
        // the objective vector and the fitness of the current solution
        ObjectiveVector x_objVec;
        double x_fitness;
        // the index, the objective vector and the fitness of the worst solution in the population (-1 implies that the worst is the newly created one)
        int worst_idx;
        ObjectiveVector worst_objVec;
        double worst_fitness;
////////////////////////////////////////////
        // the index of the extreme non-dominated points
        int ext_0_idx, ext_1_idx;
        unsigned int ind;
////////////////////////////////////////////
        // the index current of the current solution to be explored
        unsigned int i=0;
        // initilization of the move for the first individual
        moveInit(move, _pop[i]);
        while (i<_pop.size() && continuator(_pop))
        {
            // x = one neigbour of pop[i]
            // evaluate x in the objective space
            x_objVec = moveIncrEval(move, _pop[i]);
            // update every fitness values to take x into account and compute the fitness of x
            x_fitness = fitnessAssignment.updateByAdding(_pop, x_objVec);

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // extremes solutions
            OneObjectiveComparator comp0(0);
            ext_0_idx = std::min_element(_pop.begin(), _pop.end(), comp0) - _pop.begin();
            OneObjectiveComparator comp1(1);
            ext_1_idx = std::min_element(_pop.begin(), _pop.end(), comp1) - _pop.begin();
            // new = extreme ?
            if (x_objVec[0] < _pop[ext_0_idx].objectiveVector()[0])
            {
                ext_0_idx = -1;
            }
            else if ( (x_objVec[0] == _pop[ext_0_idx].objectiveVector()[0]) && (x_objVec[1] < _pop[ext_0_idx].objectiveVector()[1]) )
            {
                ext_0_idx = -1;
            }
            else if (x_objVec[1] < _pop[ext_1_idx].objectiveVector()[1])
            {
                ext_1_idx = -1;
            }
            else if ( (x_objVec[1] == _pop[ext_1_idx].objectiveVector()[1]) && (x_objVec[0] < _pop[ext_1_idx].objectiveVector()[0]) )
            {
                ext_1_idx = -1;
            }
            // worst init
            if (ext_0_idx == -1)
            {
                ind = 0;
                while (ind == ext_1_idx)
                {
                    ind++;
                }
                worst_idx = ind;
                worst_objVec = _pop[ind].objectiveVector();
                worst_fitness = _pop[ind].fitness();
            }
            else if (ext_1_idx == -1)
            {
                ind = 0;
                while (ind == ext_0_idx)
                {
                    ind++;
                }
                worst_idx = ind;
                worst_objVec = _pop[ind].objectiveVector();
                worst_fitness = _pop[ind].fitness();
            }
            else
            {
                worst_idx = -1;
                worst_objVec = x_objVec;
                worst_fitness = x_fitness;
            }
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // who is the worst ?
            for (unsigned int j=0; j<_pop.size(); j++)
            {
                if ( (j!=ext_0_idx) && (j!=ext_1_idx) )
                {
                    if (_pop[j].fitness() < worst_fitness)
                    {
                        worst_idx = j;
                        worst_objVec = _pop[j].objectiveVector();
                        worst_fitness = _pop[j].fitness();
                    }
                }
            }
            // if the worst solution is the new one
            if (worst_idx == -1)
            {
                // if all its neighbours have been explored,
                // let's explore the neighborhoud of the next individual
                if (! nextMove(move, _pop[i]))
                {
                    i++;
                    if (i<_pop.size())
                    {
                        // initilization of the move for the next individual
                        moveInit(move, _pop[i]);
                    }
                }
            }
            // if the worst solution is located before _pop[i]
            else if (worst_idx <= i)
            {
                // the new solution takes place insteed of _pop[worst_idx]
                _pop[worst_idx] = _pop[i];
                move(_pop[worst_idx]);
                _pop[worst_idx].objectiveVector(x_objVec);
                _pop[worst_idx].fitness(x_fitness);
                // let's explore the neighborhoud of the next individual
                i++;
                if (i<_pop.size())
                {
                    // initilization of the move for the next individual
                    moveInit(move, _pop[i]);
                }
            }
            // if the worst solution is located after _pop[i]
            else if (worst_idx > i)
            {
                // the new solution takes place insteed of _pop[i+1] and _pop[worst_idx] is deleted
                _pop[worst_idx] = _pop[i+1];
                _pop[i+1] = _pop[i];
                move(_pop[i+1]);
                _pop[i+1].objectiveVector(x_objVec);
                _pop[i+1].fitness(x_fitness);
                // let's explore the neighborhoud of the individual _pop[i+2]
                i += 2;
                if (i<_pop.size())
                {
                    // initilization of the move for the next individual
                    moveInit(move, _pop[i]);
                }
            }
            // update fitness values
            fitnessAssignment.updateByDeleting(_pop, worst_objVec);
        }
    }






//////////////////////////////////////////////////////////////////////////////////////////
class OneObjectiveComparator : public moeoComparator < MOEOT >
    {
    public:
        OneObjectiveComparator(unsigned int _obj) : obj(_obj)
        {
            if (obj > MOEOT::ObjectiveVector::nObjectives())
            {
                throw std::runtime_error("Problem with the index of objective in OneObjectiveComparator");
            }
        }
        const bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
        {
            if (_moeo1.objectiveVector()[obj] < _moeo2.objectiveVector()[obj])
            {
                return true;
            }
            else
            {
                return (_moeo1.objectiveVector()[obj] == _moeo2.objectiveVector()[obj]) && (_moeo1.objectiveVector()[(obj+1)%2] < _moeo2.objectiveVector()[(obj+1)%2]);
            }
        }
    private:
        unsigned int obj;
    };
//////////////////////////////////////////////////////////////////////////////////////////




};

#endif /*MOEOIBMOLS_H_*/
