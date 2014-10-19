// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSGA.h
// (c) Marc.Schoenauer 2000 - Maarten Keijzer 2000
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
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoSGATransform_h
#define _eoSGATransform_h

#include <eoInvalidateOps.h>
#include <eoPop.h>

///////////////////////////////////////////////////////////////////////////////
// class eoSGATransform
///////////////////////////////////////////////////////////////////////////////
#include <vector>          // std::vector
#include <utils/eoRNG.h>
#include <eoTransform.h>

/** eoSGATransform: transforms a population using genetic operators.
 * It does it exactly as class eoSGA, i.e. only accepts
 *    quadratic crossover and unary mutation
 * It is here mainly for tutorial reasons
 *
 * @ingroup Combination
*/
template<class EOT> class eoSGATransform : public eoTransform<EOT>
{
 public:

  /// Default constructor.
  eoSGATransform(eoQuadOp<EOT>& _cross, double _cProba,
                 eoMonOp<EOT>& _mutate, double _mProba)
    : cross(_cross),
      crossoverProba(_cProba),
      mutate(_mutate),
      mutationProba(_mProba) {}


  /**
   * Transforms a population.
   * @param _pop The population to be transformed.
   */
  void operator()(eoPop<EOT>& _pop)
  {
    unsigned i;

    for (i=0; i<_pop.size()/2; i++)
      {
        if ( rng.flip(crossoverProba) )
          {
            // this crossover generates 2 offspring from two parents
            cross(_pop[2*i], _pop[2*i+1]);
          }
      }

    for (i=0; i < _pop.size(); i++)
      {
        if (rng.flip(mutationProba) )
          {
            mutate(_pop[i]);
          }

      }
  };

 private:
  eoInvalidateQuadOp<EOT> cross;
  double crossoverProba;
  eoInvalidateMonOp<EOT> mutate;
  double mutationProba;
};

/** eoDynSGATransform: transforms a population using genetic operators.
 * It is the Dynamic version of the above eoSGATransform
 *    i.e. the operators probabilities can be passed as an eoValueParam,
 *    and hence can be modified from outside
 * It is here mainly for tutorial reasons
 *
 * @ingroup Combination
*/
template<class EOT> class eoDynSGATransform : public eoTransform<EOT>
{
 public:

  /// Default constructor - receives values
  eoDynSGATransform(eoQuadOp<EOT>& _cross, double _cProba,
                 eoMonOp<EOT>& _mutate, double _mProba)
    : cross(_cross),
      crossoverProbaHolder(_cProba), crossoverProba(crossoverProbaHolder),
      mutate(_mutate),
      mutationProbaHolder(_mProba), mutationProba(mutationProbaHolder) {}

  /// This constructor receives pointers
  //  these will usually be some eoValueParam<double>.value()
  //  hence the ...Holder data will bever be used in this case
  eoDynSGATransform(eoQuadOp<EOT>& _cross, double* _cProbaRef,
                 eoMonOp<EOT>& _mutate, double* _mProbaRef)
    : cross(_cross),
      crossoverProbaHolder(0), crossoverProba(*_cProbaRef),
      mutate(_mutate),
      mutationProbaHolder(0), mutationProba(*_mProbaRef) {}


  /**
   * Transforms a population.
   * @param _pop The population to be transformed.
   */
  void operator()(eoPop<EOT>& _pop)
  {
    unsigned i;

    for (i=0; i<_pop.size()/2; i++)
      {
        if ( rng.flip(crossoverProba) )
          {
            // this crossover generates 2 offspring from two parents
            cross(_pop[2*i], _pop[2*i+1]);
          }
      }

    for (i=0; i < _pop.size(); i++)
      {
        if (rng.flip(mutationProba) )
          {
            mutate(_pop[i]);
          }

      }
  };
  // accessors - mainly for EASEA
  double & PCrossHandle() { return crossoverProba;}
  double & PMutHandle() { return mutationProba;}

private:
  // difference with eoSGATransform: the operator probabilities
  // they can be passed by reference or by value.
  // hence we need here to use a reference, and to eventually store a value
  eoInvalidateQuadOp<EOT> cross;
  double crossoverProbaHolder;	// the value, used only if ctor gets a value
  double& crossoverProba;       // the reference, to be used in operator()
  eoInvalidateMonOp<EOT> mutate;
  double mutationProbaHolder;	// the value, used only if ctor gets a value
  double& mutationProba;        // the reference, to be used in operator()
};


#endif
