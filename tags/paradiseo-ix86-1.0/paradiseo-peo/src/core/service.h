// "service.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __service_h
#define __service_h

#include "communicable.h"
#include "thread.h"

typedef unsigned SERVICE_ID;

class Service : public Communicable {

public :

  void setOwner (Thread & __owner);
  
  Thread * getOwner (); 

  void requestResourceRequest (unsigned __how_many = 1);
  void packResourceRequest ();

  virtual void packData ();
  virtual void unpackData ();

  virtual void execute ();
  
  virtual void packResult ();
  virtual void unpackResult ();

  virtual void notifySendingData ();
  virtual void notifySendingResourceRequest ();
  virtual void notifySendingAllResourceRequests ();

private :

  Thread * owner; /* Owner thread (i.e. 'uses' that service) */ 

  unsigned num_sent_rr; /* Number of RR not really sent (i.e. still in the sending queue)*/

};

extern Service * getService (SERVICE_ID __key); 

#endif
