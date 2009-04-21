/*
* <moeoIBMOLS.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
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
#include <moeoPopLS.h>
#include <archive/moeoArchive.h>
#include <archive/moeoUnboundedArchive.h>
#include <fitness/moeoBinaryIndicatorBasedFitnessAssignment.h>
#include <moMoveIncrEval.h>

/**
 * Indicator-Based Multi-Objective Local Search (IBMOLS) as described in
 * Basseur M., Burke K. : "Indicator-Based Multi-Objective Local Search" (2007).
 */
template < class MOEOT, class Move >
class moeoIBMOLS : public moeoPopLS < Move>
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
     * @param _arch the archive
     */
    moeoIBMOLS(
      moMoveInit < Move > & _moveInit,
      moNextMove < Move > & _nextMove,
      eoEvalFunc < MOEOT > & _eval,
      moMoveIncrEval < Move , ObjectiveVector > & _moveIncrEval,
      moeoBinaryIndicatorBasedFitnessAssignment < MOEOT > & _fitnessAssignment,
      eoContinue < MOEOT > & _continuator,
      moeoArchive < MOEOT > & _arch
    ) :
        moveInit(_moveInit),
        nextMove(_nextMove),
        eval(_eval),
        moveIncrEval(_moveIncrEval),
        fitnessAssignment (_fitnessAssignment),
        continuator (_continuator),
        arch(_arch)
    {}


    /**
     * Apply the local search until a local archive does not change or
     * another stopping criteria is met and update the archive _arch with new non-dominated solutions.
     * @param _pop the initial population
     */
    void operator() (eoPop < MOEOT > & _pop)
    {
      // evaluation of the objective values

              for (unsigned int i=0; i<_pop.size(); i++)
              {
                  eval(_pop[i]);
              }

      // fitness assignment for the whole population
      fitnessAssignment(_pop);
      // creation of a local archive
      moeoUnboundedArchive < MOEOT > archive;
      // creation of another local archive (for the stopping criteria)
      moeoUnboundedArchive < MOEOT > previousArchive;
      // update the archive with the initial population
      archive(_pop);
      do
        {
          previousArchive(archive);
          oneStep(_pop);
          archive(_pop);
        }
      while ( (! archive.equals(previousArchive)) && (continuator(arch)) );
      arch(archive);
    }


  private:

    /** the move initializer */
    moMoveInit < Move > & moveInit;
    /** the neighborhood explorer */
    moNextMove < Move > & nextMove;
    /** the full evaluation */
    eoEvalFunc < MOEOT > & eval;
    /** the incremental evaluation */
    moMoveIncrEval < Move, ObjectiveVector > & moveIncrEval;
    /** the fitness assignment strategy */
    moeoBinaryIndicatorBasedFitnessAssignment < MOEOT > & fitnessAssignment;
    /** the stopping criteria */
    eoContinue < MOEOT > & continuator;
    /** archive */
    moeoArchive < MOEOT > & arch;

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
//          ext_0_idx = -1;
//          ext_0_objVec = x_objVec;
//          ext_1_idx = -1;
//          ext_1_objVec = x_objVec;
//          for (unsigned int k=0; k<_pop.size(); k++)
//            {
//              // ext_0
//              if (_pop[k].objectiveVector()[0] < ext_0_objVec[0])
//                {
//                  ext_0_idx = k;
//                  ext_0_objVec = _pop[k].objectiveVector();
//                }
//              else if ( (_pop[k].objectiveVector()[0] == ext_0_objVec[0]) && (_pop[k].objectiveVector()[1] < ext_0_objVec[1]) )
//                {
//                  ext_0_idx = k;
//                  ext_0_objVec = _pop[k].objectiveVector();
//                }
//              // ext_1
//              else if (_pop[k].objectiveVector()[1] < ext_1_objVec[1])
//                {
//                  ext_1_idx = k;
//                  ext_1_objVec = _pop[k].objectiveVector();
//                }
//              else if ( (_pop[k].objectiveVector()[1] == ext_1_objVec[1]) && (_pop[k].objectiveVector()[0] < ext_1_objVec[0]) )
//                {
//                  ext_1_idx = k;
//                  ext_1_objVec = _pop[k].objectiveVector();
//                }
//            }
//          // worst init
//          if (ext_0_idx == -1)
//            {
//              ind = 0;
//              while (ind == ext_1_idx)
//                {
//                  ind++;
//                }
//              worst_idx = ind;
//              worst_objVec = _pop[ind].objectiveVector();
//              worst_fitness = _pop[ind].fitness();
//            }
//          else if (ext_1_idx == -1)
//            {
//              ind = 0;
//              while (ind == ext_0_idx)
//                {
//                  ind++;
//                }
//              worst_idx = ind;
//              worst_objVec = _pop[ind].objectiveVector();
//              worst_fitness = _pop[ind].fitness();
//            }
//          else
//            {
//              worst_idx = -1;
//              worst_objVec = x_objVec;
//              worst_fitness = x_fitness;
//            }
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
