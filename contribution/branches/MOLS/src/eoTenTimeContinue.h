#ifndef _eoTenTimeContinue_h
#define _eoTenTimeContinue_h

#include <time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <eoContinue.h>

template< class EOT>
class eoTenTimeContinue: public eoContinue<EOT>
{
public:

    eoTenTimeContinue(unsigned int _maxTime, std::string _fileName, moeoArchive<EOT> & _arch) :
            start(time(0)), maxTime(_maxTime), id(1), fileName(_fileName), arch(_arch) {}


    // _pop must be an archive
    virtual bool operator() (const eoPop<EOT> & _pop)
    {
        unsigned int diff = (unsigned int) difftime(time(0), start);
        if (diff >= (id * maxTime/10) )
        {
            time_t begin=time(0);
            save(_pop);
            id++;
            start= start - (time(0)-begin);
	    //operator()(_pop);
        }
        if (diff >= maxTime)
        {
            return false;
        }
        return true;
    }


    virtual std::string className(void) const
    {
        return "eoTenTimeContinue";
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

    time_t start;
    unsigned int maxTime;
    unsigned int id;
    std::string fileName;
    moeoArchive<EOT> & arch;


    void save(const eoPop<EOT> & _pop)
    {
        // update the archive
        arch(_pop);
        // save the archive contents in a file
        std::string tmp = fileName;
        std::ostringstream os;
        os << id;
        tmp += '.';
        tmp += os.str();
        std::ofstream outfile(tmp.c_str());
// std::cout << "save " << tmp << " - " << difftime(time(0), start) << std::endl;
        unsigned int nObj = EOT::ObjectiveVector::nObjectives();
        for (unsigned int i=0; i<arch.size(); i++)
        {
            for (unsigned int j=0; j<nObj; j++)
            {
                outfile << arch[i].objectiveVector()[j];
                if (j != nObj -1)
                {
                    outfile << ' ';
                }
            }
            outfile << std::endl;
        }
        outfile.close();
    }

};

#endif
