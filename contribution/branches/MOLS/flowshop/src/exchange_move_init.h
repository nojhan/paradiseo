#ifndef exchange_move_init_h
#define exchange_move_init_h

#include <moMoveInit.h>

//#include <moeo>

#include "exchange_move.h"

/** It sets the first couple of edges */
class ExchangeMoveInit : public moMoveInit <ExchangeMove>
  {

  public :

    void operator () (ExchangeMove & _move, const FlowShop & _flowshop) ;

  } ;

#endif
