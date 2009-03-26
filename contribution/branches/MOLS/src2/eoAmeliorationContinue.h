#ifndef _eoAmeliorationContinue_h
#define _eoAmeliorationContinue_h

#include <iostream>
#include <sstream>
#include <fstream>
#include <eoContinue.h>
#include <moeoDMLSArchive.h>
#include <archive/moeoArchive.h>

template< class EOT>
class eoAmeliorationContinue: public eoContinue<EOT>
{
public:

    eoAmeliorationContinue(moeoDMLSArchive<EOT> & _arch, unsigned int _neighborhoodSize, bool _multiply) : arch(_arch),maxGen(_neighborhoodSize), neighborhoodSize(_neighborhoodSize), counter(0), multiply(_multiply){}

    // _pop must be an archive
    virtual bool operator() (const eoPop<EOT> & _pop)
    {
    	bool res;
    	if(multiply){
    		maxGen=arch.size() * neighborhoodSize;
    	}
    	else{
    		maxGen = neighborhoodSize;
    	}
    	if(arch.modified())
    		counter=0;
    	else
    		counter++;
    	res = (counter < maxGen);
    	if(!res)
    		counter=0;
    	return res;
    }
   
    virtual std::string className(void) const
    {
        return "eoAmeliorationContinue";
    }

private:

	moeoDMLSArchive <EOT> & arch;
    unsigned int maxGen;
    unsigned int neighborhoodSize;
    unsigned int counter;
    bool multiply;

};

#endif
