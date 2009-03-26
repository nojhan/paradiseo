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

    eoAmeliorationContinue(moeoDMLSArchive<EOT> & _arch, unsigned int _maxGen) : arch(_arch), maxGen(_maxGen), counter(0){}

    // _pop must be an archive
    virtual bool operator() (const eoPop<EOT> & _pop)
    {
    	std::cout << counter << std::endl;
    	if(arch.modified())
    		counter=0;
    	else
    		counter++;
    	return (counter < maxGen);
    }
   
    virtual std::string className(void) const
    {
        return "eoAmeliorationContinue";
    }

private:

	moeoDMLSArchive <EOT> & arch;
    unsigned int maxGen;
    unsigned int counter;

};

#endif
