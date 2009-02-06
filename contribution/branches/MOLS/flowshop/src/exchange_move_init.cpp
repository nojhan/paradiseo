#include <exchange_move_init.h>
#include <utils/eoRNG.h>

void ExchangeMoveInit :: operator () (ExchangeMove & _move, const FlowShop & _flowshop)
{
  FlowShop flowshop=_flowshop;
  //eoRNG rng;
  do
  {
  _move.first = rng.random(flowshop.size()) ;
  _move.second = rng.random(flowshop.size()) ;
  }while(_move.first == _move.second);
}
