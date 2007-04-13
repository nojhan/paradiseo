// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoArchiveObjectiveVectorSavingUpdater.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
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
#include <moeoArchive.h>

#define MAX_BUFFER_SIZE 1000

/**
 * This class allows to save the fitnesses of solutions contained in an archive into a file at each generation.
 */
template <class EOT>
class moeoArchiveObjectiveVectorSavingUpdater : public eoUpdater
{
public:

    /**
     * Ctor
     * @param _arch local archive
     * @param _filename target filename
     * @param _id own ID
     */
    moeoArchiveObjectiveVectorSavingUpdater (moeoArchive<EOT> & _arch, const std::string & _filename, int _id = -1) : arch(_arch), filename(_filename), id(_id), counter(0)
    {}

    /**
     * Saves the fitness of the archive's members into the file
     */
    void operator()() {
        char buff[MAX_BUFFER_SIZE];
        if (id == -1)
            sprintf (buff, "%s.%u", filename.c_str(), counter ++);
        else
            sprintf (buff, "%s.%u.%u", filename.c_str(), id, counter ++);
        std::ofstream f(buff);
        for (unsigned i = 0; i < arch.size (); i++)
            f << arch[i].objectiveVector() << std::endl;
        f.close ();
    }


private:

    /** local archive */
    moeoArchive<EOT> & arch;
    /** target filename */
    std::string filename;
    /** own ID */
    int id;
    /** counter */
    unsigned counter;

};

#endif /*MOEOARCHIVEOBJECTIVEVECTORSAVINGUPDATER_H_*/
