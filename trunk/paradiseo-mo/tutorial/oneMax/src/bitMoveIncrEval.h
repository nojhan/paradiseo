// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "incrEval_funcNK.h"

#ifndef __incr_eval_funcNK_h
#define __incr_eval_funcNK_h

#include <move/moMoveIncrEval.h>
#include "bitMove.h"

template <class EOT>
class OneMaxIncrEval : public moMoveIncrEval < BitMove<EOT> > {

public :  
  OneMaxIncrEval(){ };

  typename EOT::Fitness operator () (const BitMove<EOT> & move, const EOT & chrom) {
	if(chrom[move.bit]==0){
	return chrom.fitness()+1;
}
	else{
	return chrom.fitness()-1;
}  
};
};

#endif
