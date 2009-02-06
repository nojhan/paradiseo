#ifndef exchange_move_h
#define exchange_move_h

#include <eoPersistent.h>

#include <utility>
#include <moMove.h>
//#include <moeo>

#include <FlowShop.h>

class ExchangeMove : public moMove <FlowShop>, public std :: pair <unsigned, unsigned>, public eoPersistent
  {

  public :
	   

    ExchangeMove operator ! () const ;

    //void operator () (FlowShop & __FlowShop) ;
	void operator () (FlowShop & __FlowShop) ;

    void readFrom (std :: istream & __is) ;

    void printOn (std :: ostream & __os) const ;
  } ;

#endif
