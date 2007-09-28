// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moComparator.h"

// (c) OPAC Team, LIFL, 2003-2007

/* TEXT LICENCE
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moComparator_h
#define __moComparator_h


//! Template for classes which need to compare two EOT and indicate if the first is "better" than the second. 
/*!
  The objects that extend this template describe how an EOT is "better" than an other.
 */
template<class EOT>
class moComparator: public eoBF<const EOT&, const EOT&, bool>
{
};

#endif
