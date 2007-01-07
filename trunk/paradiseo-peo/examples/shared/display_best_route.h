// "display_best_route.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __display_best_route_h
#define __display_best_route_h

#include <utils/eoUpdater.h>

#include <eoPop.h>

#include "route.h"

class DisplayBestRoute : public eoUpdater {
  
public :

  DisplayBestRoute (eoPop <Route> & __pop);
  
  void operator () ();

private :
  
  eoPop <Route> & pop;

};

#endif
