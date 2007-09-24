// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoArchiveObjectiveVectorSavingUpdater.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOARCHIVEOBJECTIVEVECTORSAVINGUPDATER_H_
#define MOEOARCHIVEOBJECTIVEVECTORSAVINGUPDATER_H_

#include <fstream>
#include <string>
#include <eoPop.h>
#include <utils/eoUpdater.h>
#include <archive/moeoArchive.h>

#define MAX_BUFFER_SIZE 1000

/**
 * This class allows to save the objective vectors of the solutions contained in an archive into a file at each generation.
 */
template < class MOEOT >
class moeoArchiveObjectiveVectorSavingUpdater : public eoUpdater
{
public:

    /**
     * Ctor
     * @param _arch local archive
     * @param _filename target filename
     * @param _count put this variable to true if you want a new file to be created each time () is called and to false if you only want the file to be updated
     * @param _id own ID
     */
    moeoArchiveObjectiveVectorSavingUpdater (moeoArchive<MOEOT> & _arch, const std::string & _filename, bool _count = false, int _id = -1) :
            arch(_arch), filename(_filename), count(_count), counter(0), id(_id)
    {}


    /**
     * Saves the fitness of the archive's members into the file
     */
    void operator()() {
        char buff[MAX_BUFFER_SIZE];
        if (count)
        {
            if (id == -1)
            {
                sprintf (buff, "%s.%u", filename.c_str(), counter ++);
            }
            else
            {
                sprintf (buff, "%s.%u.%u", filename.c_str(), id, counter ++);
            }
        }
        else
        {
            if (id == -1)
            {
                sprintf (buff, "%s", filename.c_str());
            }
            else
            {
                sprintf (buff, "%s.%u", filename.c_str(), id);
            }
            counter ++;
        }
        std::ofstream f(buff);
        for (unsigned int i = 0; i < arch.size (); i++)
            f << arch[i].objectiveVector() << std::endl;
        f.close ();
    }


private:

    /** local archive */
    moeoArchive<MOEOT> & arch;
    /** target filename */
    std::string filename;
    /** this variable is set to true if a new file have to be created each time () is called and to false if the file only HAVE to be updated */
    bool count;
    /** counter */
    unsigned int counter;
    /** own ID */
    int id;

};

#endif /*MOEOARCHIVEOBJECTIVEVECTORSAVINGUPDATER_H_*/
