// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEvalFunc.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef eoEvalFunc_H
#define eoEvalFunc_H

/** Evaluate: takes one EO and sets its "fitness" property
 * returning this fitness also. That is why EOT is passed by
 * non-const reference: it must be altered within evaluate.\\

 The requirements on the types with which this class is to be
 instantiated with are null, or else, they depend on the particular
 class it's going to be applied to; EO does not impose any requirement 
 on it. If you subclass this abstract class, and use it to evaluate an 
 EO, the requirements on this EO will depend on the evaluator.
 */
template< class EOT >
struct eoEvalFunc {

#ifdef _MSC_VER
	typedef EOT::Fitness EOFitT;
#else
	typedef typename Fitness::EOFitT EOFitT;
#endif

  /// Effectively applies the evaluation function to an EO or urEO
  virtual EOFitT evaluate( EOT & _eo ) const = 0;
};

#endif