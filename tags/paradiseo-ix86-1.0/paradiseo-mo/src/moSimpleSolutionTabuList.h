/* <moSimpleSolutionTabuList.h>  
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

#ifndef __moSimpleSolutionTabuList_h
#define __moSimpleSolutionTabuList_h

#include <list>
#include <iterator>

#include "moTabuList.h"

//! Class describing a solution tabu list with limited length.
template <class M>
class moSimpleSolutionTabuList: public moTabuList < M >
{
  
public:

  //! Alias for the type
  typedef typename M::EOType EOT;
  
  //! Constructor
  /*!
    \param __size The maximum size of the solution tabu list.
   */
  moSimpleSolutionTabuList(unsigned int __size): maxSize(__size)
  {
    currentSize=0;
  }
  
  //! Function that indicates if, in a given state, the _move is tabu or not.
  /*!
    \param __move A given moMove.
    \param __sol A solution.
    \return true or false.
  */
  bool operator () (const M & __move, const EOT & __sol) 
  {
    typename std::list<EOT>::iterator it;
  
    M _move=(M)__move;
    EOT _sol=(EOT) __sol;

    _move(_sol);
  
    it=tabuList.begin();
    while(it!=tabuList.end()&&(!((*it)==_sol)))
      {
	it++;
      }
    
    return it!=tabuList.end();
  }
  
  void
  add (const M & __move, const EOT & __sol)
  {
    M _move=(M)__move;
    EOT _sol=(EOT) _sol;
    
    _move(_sol);

    if(currentSize!=0)
      {
	// Useful in the case of a solution has been kept thanks to the moAspirCrit.
	// In this case, the solution can already be in the tabuList.
	removeSolution(_sol);
      }
    
    tabuList.push_back(_sol);
    
    if(currentSize==maxSize)
      {
	tabuList.erase(tabuList.begin());
      }
    else
      {
	currentSize++;
      }
  }

  void
  update ()
  {
    //nothing to do
  }

  void
  init ()
  {
    //nothing to do
  }

private:

  //! Procedure that removes a given solution from the tabu list (if it is into, else does nothing).
  /*!
    \param __sol A given solution.
  */
  void
  removeSolution(const EOT & __sol)
  {
    typename std::list<EOT>::iterator it;

    it=tabuList.begin();
    while(it!=tabuList.end()&&(!((*it)==__sol)))
      {
	it++;
      }

    if(it!=tabuList.end())
      {
	tabuList.erase(it);
      }
  }
  
  //! The maximum size of the tabu list.
  unsigned int maxSize;

  //! The current size of the tabu list.
  unsigned int currentSize;
  
  //! The solution tabu list.
  std::list<EOT> tabuList;
};

#endif
