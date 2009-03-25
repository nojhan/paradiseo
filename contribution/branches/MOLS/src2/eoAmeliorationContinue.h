#ifndef _eoAmeliorationContinue_h
#define _eoAmeliorationContinue_h

#include <iostream>
#include <sstream>
#include <fstream>
#include <eoContinue.h>

template< class EOT>
class eoAmeliorationContinue: public eoContinue<EOT>
{
public:

    eoAmeliorationContinue(unsigned int _maxGen) : maxGen(_maxGen), counter(0){}


    // _pop must be an archive
    virtual bool operator() (const eoPop<EOT> & _pop)
    {
    	if(_pop.modified())
    		counter=0;
    	else
    		counter++;
    	return (counter < maxGen);
    }


    virtual std::string className(void) const
    {
        return "eoAmeliorationContinue";
    }


    void readFrom (std :: istream & __is)
    {

        __is >> start;
    }


    void printOn (std :: ostream & __os) const
    {

        __os << start << ' ' << std :: endl;
    }

private:

    unsigned int maxGeneration;
    unsigned int counter;

};

#endif
