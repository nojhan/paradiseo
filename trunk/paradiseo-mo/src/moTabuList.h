/*
* <moTabuList.h>
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

#ifndef __moTabuList_h
#define __moTabuList_h

#include <eoFunctor.h>

//! Class describing a tabu list that a moTS uses.
/*!
  It is only a description, does nothing... A new object that herits from this class has to be defined in order
  to be used in a moTS.
 */
template < class M > class moTabuList:public eoBF < const M &, const typename
      M::EOType &,
      bool >
  {

  public:
    //! Alias for the type
    typedef typename M::EOType EOT;

    //! Procedure to add a move in the tabu list
    /*!
       The two parameters have not to be modified so they are constant parameters.

       \param __move a new tabu move.
       \param __sol the origianl solution associated to this move.
     */
    virtual void
    add (const M & __move, const EOT & __sol) = 0;

    //! Procedure that updates the tabu list content.
    /*!
       Generally, a counter associated to each saved move is decreased by one.
    */
    virtual void
    update () = 0;

    //! Procedure which initialises the tabu list.
    /*!
       Can be useful if the data structure needs to be allocated before being used.
     */
    virtual void
    init () = 0;
  };

#endif
