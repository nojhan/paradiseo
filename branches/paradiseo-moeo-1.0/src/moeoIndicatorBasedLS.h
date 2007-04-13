// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoIndicatorBasedLS.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOINDICATORBASEDLS_H_
#define MOEOINDICATORBASEDLS_H_

#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoPop.h>
#include <moMove.h>
#include <moMoveInit.h>
#include <moNextMove.h>
#include <moeoMoveIncrEval.h>
#include <moeoArchive.h>
#include <moeoLS.h>
#include <moeoIndicatorBasedFitnessAssignment.h>

/**
 * Indicator-Based Multi-Objective Local Search (IBMOLS) as described in
 * Basseur M., Burke K. : "Indicator-Based Multi-Objective Local Search" (2007).
 */
template < class MOEOT, class Move >
class moeoIndicatorBasedLS : public moeoLS < MOEOT, eoPop < MOEOT > & >
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
    moeoIndicatorBasedLS(
        moMoveInit < Move > & _moveInit,
        moNextMove < Move > & _nextMove,
        eoEvalFunc < MOEOT > & _eval,
        moeoMoveIncrEval < Move > & _moveIncrEval,
        moeoIndicatorBasedFitnessAssignment < MOEOT > & _fitnessAssignment,
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
        for (unsigned i=0; i<_pop.size(); i++)
        {
            eval(_pop[i]);
        }
        // fitness assignment for the whole population
        fitnessAssignment(_pop);
        // creation of a local archive
        moeoArchive < MOEOT > archive;
        // creation of another local archive (for the stopping criteria)
        moeoArchive < MOEOT > previousArchive;
        // update the archive with the initial population
        archive.update(_pop);
        unsigned counter=0;
        do
        {
            previousArchive.update(archive);
            oneStep(_pop);
            archive.update(_pop);
            counter++;
        } while ( (! archive.equals(previousArchive)) && (continuator(_arch)) );
        _arch.update(archive);
        cout << "\t=> " << counter << " step(s)" << endl;
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
    moeoIndicatorBasedFitnessAssignment < MOEOT > & fitnessAssignment;
    /** the stopping criteria */
    eoContinue < MOEOT > & continuator;


    /**
     * Apply one step of the local search to the population _pop
     * @param _pop the population
     */
    void oneStep (eoPop < MOEOT > & _pop)
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
        // the index current of the current solution to be explored
        unsigned i=0;
        // initilization of the move for the first individual
        moveInit(move, _pop[i]);
        while (i<_pop.size() && continuator(_pop))
        {
            // x = one neigbour of pop[i]
            // evaluate x in the objective space
            x_objVec = moveIncrEval(move, _pop[i]);
            // update every fitness values to take x into account and compute the fitness of x
            x_fitness = fitnessAssignment.updateByAdding(_pop, x_objVec);
            // who is the worst individual ?
            worst_idx = -1;
            worst_objVec = x_objVec;
            worst_fitness = x_fitness;
            for (unsigned j=0; j<_pop.size(); j++)
            {
                if (_pop[j].fitness() < worst_fitness)
                {
                    worst_idx = j;
                    worst_objVec = _pop[j].objectiveVector();
                    worst_fitness = _pop[j].fitness();
                }
            }
            // the worst solution is the new one
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
            // the worst solution is located before _pop[i]
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
            // the worst solution is located after _pop[i]
            else if (worst_idx > i)
            {
                // the new solution takes place insteed of _pop[i+1] and _pop[worst_idx] is deleted
                _pop[worst_idx] = _pop[i+1];
                _pop[i+1] = _pop[i];
                move(_pop[i+1]);
                _pop[i+1].objectiveVector(x_objVec);
                _pop[i+1].fitness(x_fitness);
                // do not explore the neighbors of the new solution immediately
                i = i+2;
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

};

#endif /*MOEOINDICATORBASEDLS_H_*/
