#include "exchange_move.h"

/*ExchangeMove ExchangeMove :: operator ! () const
  {
    ExchangeMove move = * this ;
	FlowShop __flowshop;
    //std :: swap (move.first, move.second) ;
	__flowshop[move.second] =__flowshop[move.first];
    return move ;
  }*/

//void ExchangeMove :: operator () (FlowShop & __flowshop)
void ExchangeMove :: operator () (FlowShop&  __flowshop)
{
	//bool isModified;
	//ExchangeMove move;
    int direction;
    unsigned int tmp;
    //FlowShop result = __flowshop;

 /*std :: vector <unsigned int> seq_cities ;

  for (unsigned int i = second ; i > first ; i --)
    {
      seq_cities.push_back (__flowshop [i]) ;
    }

  unsigned int j = 0 ;
  for (unsigned int i = first + 1 ; i < second + 1 ; i ++)
    {
      __flowshop[i] = seq_cities [j ++] ;
    }*/
	if (first < second)
        direction = 1;
    else
        direction = -1;
    // mutation
    tmp = __flowshop[first];
    for (unsigned int i=first ; i!= second ; i+=direction)
        __flowshop[i] = __flowshop[i+direction];
    __flowshop[ second] = tmp;
    // update (if necessary)
   /* if (result != __flowshop)
    {
        // update
        __flowshop.value(result);
        // the genotype has been modified
        //isModified = true;
    }
    /*else
    {
        // the genotype has not been modified
        isModified = false;
    }*/
    // return 'true' if the genotype has been modified
    //return isModified;
}

void ExchangeMove :: readFrom (std :: istream & __is)
{
  __is >> first >> second ;
}

void ExchangeMove :: printOn (std :: ostream & __os) const
  {
    __os << first << ' ' << second ;
  }
