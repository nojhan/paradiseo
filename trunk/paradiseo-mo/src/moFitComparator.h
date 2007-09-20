// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moFitComparator.h"

// (c) OPAC Team, LIFL, 2003-2007

/* TEXT LICENCE
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moFitComparator_h
#define __moFitComparator_h

//! Comparison according to the fitness. 
/*!
  An EOT is better than an other if its fitness is better.
 */
template<class EOT>
class moFitComparator: public moComparator<EOT>
{
 public:
  bool operator()(const EOT& _solution1, const EOT& _solution2)
  {
    return _solution1.fitness()>_solution2.fitness();
  }
};

#endif
