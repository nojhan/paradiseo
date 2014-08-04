/*
* <moeoArchiveObjectiveVectorSavingUpdater.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#ifndef MOEOARCHIVEOBJECTIVEVECTORSAVINGUPDATER_H_
#define MOEOARCHIVEOBJECTIVEVECTORSAVINGUPDATER_H_

#include <fstream>
#include <string>
#include "../../eo/eoPop.h"
#include "../../eo/utils/eoUpdater.h"
#include "../archive/moeoArchive.h"

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
    void operator()()
    {
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
