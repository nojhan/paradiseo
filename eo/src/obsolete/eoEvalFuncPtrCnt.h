/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

-----------------------------------------------------------------------------
    eoEvalFuncPtrCnt.h
      Same as eoEvalFuncPtr, but counts the number of evaluations

    (c) GeNeura Team, 2000
 
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
 */
//-----------------------------------------------------------------------------

#ifndef EOEVALFUNCPTRCNT_H
#define EOEVALFUNCPTRCNT_H

#include <eoEvalFuncPtr.h>

/** EOEvalFuncCntPtr: This class
 * takes an existing function pointer and converts it into a evaluation
 * function class. That way, old style C or C++ functions can be adapted to EO
 * function classes.
 */
template< class EOT >
struct eoEvalFuncPtrCnt: public eoEvalFuncPtr<EOT> {

  /** Applies the function to the chromosome and sets the fitness of the
      Chrom. Thus, the evaluation function need not be worried about that.
      @param _eval pointer to the evaluation function, takes a EOT as an
                   argument and returns the fitness
      @return the evaluated fitness for that object.
  */
  eoEvalFuncPtrCnt( EOFitT (* _eval)( const EOT& ) )
    : eoEvalFuncPtr<EOT>( _eval ), numOfEvaluations( 0 )  {};
  
  /// Effectively applies the evaluation function to an EO 
  virtual void operator() ( EOT & _eo ) const {
     eoEvalFuncPtr<EOT>::operator()( _eo );
    numOfEvaluations++;
  };

  ///
  unsigned getNumOfEvaluations() const { return numOfEvaluations; };
  
private:
  mutable unsigned numOfEvaluations;
};

#endif
