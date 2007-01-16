// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoParetoPhenDist.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#include<eoParetoFitness.h>

template < class EOT, class DistType > class moeoParetoPhenDist
{
public:
  virtual DistType operator  ()(const EOT & eopf1, const EOT & eopf2) = 0;

};



//Euclidien distance

template < class EOT, class DistType =
  double >class moeoParetoEuclidDist:public moeoParetoPhenDist < EOT,
  DistType >
{

public:
  DistType operator  () (const EOT & eopf1, const EOT & eopf2)
  {
    double res = 0.0;
    double max = 0.0;
    double temp;
    for (unsigned i = 0; i < eopf1.fitness ().size (); ++i)
      {
	temp =
	  (eopf1.fitness ().operator[](i) -
	   eopf2.fitness ().operator[](i)) * (eopf1.fitness ().operator[](i) -
					      eopf2.fitness ().operator[](i));
	if (temp > max)
	  max = temp;		/* for normalization */
	res = res + temp;
      }
    return sqrt (res / max);
  }

};
