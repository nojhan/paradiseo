// "cooperative.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __cooperative_h
#define __cooperative_h

#include "communicable.h"
#include "runner.h"

typedef unsigned COOP_ID;

class Cooperative : public Communicable {

public :

  Runner * getOwner ();

  void setOwner (Runner & __runner);

  virtual void pack () = 0;
  
  virtual void unpack () = 0;

  void send (Cooperative * __coop); 

  virtual void notifySending ();

private :

  Runner * owner;

};

extern Cooperative * getCooperative (COOP_ID __key); 

#endif
