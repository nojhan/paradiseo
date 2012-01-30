// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoParticleFullInitializer.h
// (c) OPAC Team, INRIA, 2007
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

    Contact: clive.canape@inria.fr


 */
//-----------------------------------------------------------------------------

#ifndef _eoParticleFullInitializer_H
#define _eoParticleFullInitializer_H

#include <utils/eoRealVectorBounds.h>
#include <eoVelocityInit.h>
#include <eoPop.h>
#include <eoParticleBestInit.h>
#include <eoTopology.h>

/**
 @addtogroup Initializators
 @{
 */


/*
 * Abstract class for initialization of algorithm PSO
 */
template <class POT> class eoInitializerBase : public eoFunctorBase
  {
  public :

    virtual ~eoInitializerBase()
    {}

    virtual void operator()()
    {};
  };

/**
        Base (name) class for Initialization of algorithm PSO

        @see eoInitializerBase eoUF apply
*/
template <class POT> class eoParticleInitializer : public eoInitializerBase <POT>
  {
  public:

    //!	Constructor
    //! @param _proc Evaluation function
    //! @param _initVelo Initialization of the velocity
    //! @param _initBest Initialization of the best
    //! @param _topology Topology to use
    //! @param _pop Population
    eoParticleFullInitializer(
      eoUF<POT&, void>& _proc,
      eoVelocityInit < POT > &_initVelo,
      eoParticleBestInit <POT> &_initBest,
      eoTopology <POT> &_topology,
      eoPop < POT > &_pop
    ) : proc(_proc), procPara(dummyEval), initVelo(_initVelo), initBest(_initBest), topology(_topology), pop(_pop) {}


    //!	Constructor for parallel evaluation
    //! @param _proc Evaluation function
    //! @param _initVelo Initialization of the velocity
    //! @param _initBest Initialization of the best
    //! @param _topology Topology to use
    //! @param _pop Population
    eoParticleFullInitializer(
      eoPopEvalFunc <POT>& _proc,
      eoVelocityInit < POT > &_initVelo,
      eoParticleBestInit <POT> &_initBest,
      eoTopology <POT> &_topology,
      eoPop < POT > &_pop
    ) : proc(dummy), procPara(_proc), initVelo(_initVelo), initBest(_initBest), topology(_topology), pop(_pop)
    {}


    //! Give the name of the class
    //! @return The name of the class
    virtual std::string className (void) const
      {
        return "eoInitializer";
      }



    virtual void operator  () ()
    {
                eoPop<POT> empty_pop;

                // evaluates using either the "sequential" evaluator ...
                apply(proc, pop);

                // ... or the parallel one
        procPara(empty_pop, pop);

        // no matter what is the eval operator, initializes the velocities and the particle's best
        apply < POT > (initVelo, pop);
        apply < POT > (initBest, pop);

        // finally setup the topology. We have now all we need to do so.
        topology.setup(pop);
    }

  private :

    /*
        @param proc First evaluation
        @param initVelo Initialization of the velocity
        @param initBest Initialization of the best

    */
    eoPop < POT > & pop;
    eoUF<POT&, void>& proc;
    eoPopEvalFunc <POT>& procPara;
    eoVelocityInit < POT > & initVelo;
    eoParticleBestInit <POT> & initBest;
    eoTopology <POT> & topology;
    class eoDummyEval : public eoPopEvalFunc<POT>
    {
      public:
        void operator()(eoPop<POT> &,eoPop<POT> &_pop)
        {}
    }
    dummyEval;
    class eoDummy : public eoUF<POT&, void>
    {
      public:
        void operator()(POT &)
        {}

    }
    dummy;
  };
#endif /*_eoParticleFullInitializer_H*/

/** @} */
