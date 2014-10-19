/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

The above line is usefulin Emacs-like editors
 */

/*
Template for EO objects initialization in EO
============================================
*/

#ifndef _eoOneMaxInit_h
#define _eoOneMaxInit_h

// include the base definition of eoInit
#include <paradiseo/eo/eoInit.h>

/**
 *  Always write a comment in this format before class definition
 *  if you want the class to be documented by Doxygen
 *
 * There is NO ASSUMPTION on the class GenoypeT.
 * In particular, it does not need to derive from EO (e.g. to initialize
 *    atoms of an eoVector you will need an eoInit<AtomType>)
 */
template <class GenotypeT>
class eoOneMaxInit: public eoInit<GenotypeT> {
public:
	/// Ctor - no requirement
// START eventually add or modify the anyVariable argument
//  eoOneMaxInit()
    eoOneMaxInit( unsigned  _vecSize) : vecSize(_vecSize)
// END eventually add or modify the anyVariable argument
  {
    // START Code of Ctor of an eoOneMaxInit object
    // END   Code of Ctor of an eoOneMaxInit object
  }


  /** initialize a genotype
   *
   * @param _genotype  generally a genotype that has been default-constructed
   *                   whatever it contains will be lost
   */
  void operator()(GenotypeT & _genotype)
  {
    // START Code of random initialization of an eoOneMax object
    vector<bool> b(vecSize);
    for (unsigned i=0; i<vecSize; i++)
      b[i]=rng.flip();
    _genotype.setB(b);
    // END   Code of random initialization of an eoOneMax object
    _genotype.invalidate();	   // IMPORTANT in case the _genotype is old
  }

private:
// START Private data of an eoOneMaxInit object
  unsigned vecSize;     // size of all bitstrings that this eoInit randomize
  //  varType anyVariable;		   // for example ...
// END   Private data of an eoOneMaxInit object
};

#endif
