#include "exchange_move_next.h"
//#include "graph.h"
//int ExchangeMoveNext ::init(static int i)


bool ExchangeMoveNext :: operator () (ExchangeMove & _move, const FlowShop & _flowshop)
{
  FlowShop flowshop =_flowshop;


  if (_move.first >= (flowshop.size() - 4))// && _move.second == _move.first + 2)
  //if (_move.second == flowshop.size ()-2)
    {
      return false ;
    }
  else
    {
		//std::cout<<_move.second<<" "<<_move.first<<" "<<flowshop.size()<<std::endl;
      _move.second ++ ;
      if (_move.second >= flowshop.size () - 1)
        {
          _move.first ++ ;
          _move.second = _move.first + 2 ;
        }
		

      return true ;
    }
   
}
