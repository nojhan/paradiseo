// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSGAGenOp.h
// (c) Marc.Schoenauer 2002
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
             Marc.Schoenauer@inria.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoSGAGenOp_h
#define _eoSGAGenOp_h

#include "eoGenOp.h"
#include "eoInvalidateOps.h"
///////////////////////////////////////////////////////////////////////////////
// class eoSGAGenOp
///////////////////////////////////////////////////////////////////////////////

/** ***************************************************************************
 * eoSGAGenOp (for Simple GA) mimicks the usual crossover with proba pCross + 
 * mutation with proba pMut inside an eoGeneralOp
 * It does it exactly as class eoSGATransform, i.e. only accepts 
 *    quadratic crossover and unary mutation
 * It was introduced for didactic reasons, but seems to be popular :-)
 *****************************************************************************/
template<class EOT> 
class eoSGAGenOp : public eoGenOp<EOT>
{
 public:
    
  /** Ctor from crossover (with proba) and mutation (with proba)
   * Builds the sequential op that first applies a proportional choice
   * between the crossover and nothing (cloning), then the mutation
   */
  eoSGAGenOp(eoQuadOp<EOT>& _cross, double _pCross, 
		 eoMonOp<EOT>& _mut, double _pMut)
    : cross(_cross),
      pCross(_pCross),
      mut(_mut), 
      pMut(_pMut) 
  {
    // the crossover - with probability pCross
    propOp.add(cross, pCross); // crossover, with proba pcross
    propOp.add(quadClone, 1-pCross); // nothing, with proba 1-pcross
    
    // now the sequential
    op.add(propOp, 1.0);	 // always do combined crossover
    op.add(mut, pMut);     // then mutation, with proba pmut
  }
  
  /** do the job: delegate to op */
  virtual void apply(eoPopulator<EOT>& _pop)
  {
    op.apply(_pop);
  }

  /** inherited from eoGenOp */
  virtual unsigned max_production(void) {return 2;}

  virtual string className() const {return "eoSGAGenOp";}


 private:
  eoQuadOp<EOT> &cross;   // eoInvalidateXXX take the boolean output
  double pCross;
  eoMonOp<EOT> & mut;      // of the XXX op and invalidate the EOT
  double pMut;
  eoProportionalOp<EOT> propOp;
  eoQuadCloneOp<EOT> quadClone;
  eoSequentialOp<EOT> op;
};

/** ***************************************************************************
 * eoASGAGenOp (for Almost Simple GE) mimicks proportional application of 
 * one crossover and one mutation, together with a clone operator, each one
 * with relative weights.
 * This is the other almost-standard application of variation operators 
 * (see eoSGAGenOp for the completely standard).
 *****************************************************************************/
template<class EOT> 
class eoASGAGenOp : public eoGenOp<EOT>
{
 public:
    
  /** Ctor from crossover (with proba) and mutation (with proba)
   * Builds the sequential op that first applies a proportional choice
   * between the crossover and nothing (cloning), then the mutation
   */
  eoASGAGenOp(eoQuadOp<EOT>& _cross, double _pCross, 
		 eoMonOp<EOT>& _mut, double _pMut, double _pCopy)
    : cross(_cross),
      pCross(_pCross),
      mut(_mut), 
      pMut(_pMut),
      pCopy(_pCopy)
  {
    op.add(cross, pCross); // crossover, with proba pcross
    op.add(quadClone, pCopy); // nothing, with proba pCopy
    op.add(mut, pMut);     // mutation, with proba pmut
  }
  
  /** do the job: delegate to op */
  virtual void apply(eoPopulator<EOT>& _pop)
  {
    op.apply(_pop);
  }

  /** inherited from eoGenOp */
  virtual unsigned max_production(void) {return 2;}

  virtual string className() const {return "eoASGAGenOp";}


 private:
  eoQuadOp<EOT> &cross;   // eoInvalidateXXX take the boolean output
  double pCross;
  eoMonOp<EOT> & mut;      // of the XXX op and invalidate the EOT
  double pMut;
  eoMonCloneOp<EOT> monClone;
  eoProportionalOp<EOT> op;
};


#endif
