#ifndef exchange_move_next_h
#define exchange_move_next_h

#include <moNextMove.h>
#include "exchange_move.h"

//#include <moeo>

/** It updates a couple of edges */

class ExchangeMoveNext : public moNextMove <ExchangeMove>
  {

  public :

    
	bool operator () (ExchangeMove & _move, const FlowShop & _flowshop) ;
	

  } ;

#endif
