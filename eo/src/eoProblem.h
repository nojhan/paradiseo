//-----------------------------------------------------------------------------
// eoProblem.h
// (c) GeNeura Team 1998
//-----------------------------------------------------------------------------

#ifndef EOPROBLEM_H
#define EOPROBLEM_H

//-----------------------------------------------------------------------------

template<class T> class Problem
{
 public:
  typedef T Chrom;
  typedef typename T::Gene Gene;
  typedef typename T::Fitness Fitness;
  
  virtual Fitness operator()(const Chrom& chrom) = 0;
};

//-----------------------------------------------------------------------------

#endif EOPROBLEM_H
