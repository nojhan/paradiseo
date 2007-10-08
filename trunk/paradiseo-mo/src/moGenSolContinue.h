/* 
* <moGenSolContinue.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)
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

#ifndef __moGenSolContinue_h
#define __moGenSolContinue_h

#include "moSolContinue.h"

//! One possible stop criterion for a solution-based heuristic.
/*!
  The stop criterion corresponds to a maximum number of iteration.
 */
template < class EOT > class moGenSolContinue:public moSolContinue < EOT >
{

public:

  //! Simple constructor.
  /*!
     \param __maxNumGen the maximum number of generation.
   */
  moGenSolContinue (unsigned int __maxNumGen):maxNumGen (__maxNumGen), numGen (0)
  {

  }

  //! Function that activates the stop criterion.
  /*!
     Increments the counter and returns TRUE if the
     current number of iteration is lower than the given
     maximum number of iterations.

     \param __sol the current solution.
     \return TRUE or FALSE according to the current generation number.
   */
  bool operator   () (const EOT & __sol)
  {

    return (++numGen < maxNumGen);
  }

  //! Procedure which allows to initialise the generation counter.
  /*!
     It can also be used to reset the iteration counter.
   */
  void init ()
  {

    numGen = 0;
  }

private:

  //! Iteration maximum number.
  unsigned int maxNumGen;

  //! Iteration current number.
  unsigned int numGen;
};

#endif
