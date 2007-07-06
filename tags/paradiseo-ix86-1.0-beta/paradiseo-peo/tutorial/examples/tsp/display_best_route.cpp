// "display_best_route.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "display_best_route.h"
#include "display.h"

DisplayBestRoute :: DisplayBestRoute (eoPop <Route> & __pop
				      ) : pop (__pop) {
  
  
}
  
void DisplayBestRoute :: operator () () {
  
  displayRoute (pop.best_element ());
}

