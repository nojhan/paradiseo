/* <moLSCheckPoint.h>  
 *
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * Sebastien CAHON
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
 */

#ifndef __moLSCheckPoint_h
#define __moLSCheckPoinr_h

#include <eoFunctor.h>

//! Class which allows a checkpointing system.
/*!
  Thanks to this class, at each iteration, additionnal functions can be used (and not only one).
*/
template < class M > class moLSCheckPoint:public eoBF < const M &, const typename
  M::EOType &, void >
{

public:
  //! Function which launches the checkpointing
  /*!
     Each saved function is used on the current move and the current solution.

     \param __move a move.
     \param __sol a solution.
   */
  void
  operator   () (const M & __move, const typename M::EOType & __sol)
  {

    for (unsigned int i = 0; i < func.size (); i++)
      {
	func[i]->operator   ()(__move, __sol);
      }
  }

  //! Procedure which add a new function to the function vector
  /*!
     The new function is added at the end of the vector.
     \param __f a new function to add.
   */
  void
  add (eoBF < const M &, const typename M::EOType &, void >&__f)
  {

    func.push_back (&__f);
  }

private:

  //! vector of function
  std::vector < eoBF < const
    M &, const
    typename
  M::EOType &, void >*>
    func;

};

#endif
