/* 
* <eoVector_comm.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar, Clive Canape
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

#ifndef __eoVector_mesg_h
#define __eoVector_mesg_h

#include <eoVector.h>
#include <core/moeoVector.h>

#include "messaging.h"


template <class F, class T> void pack (const eoVector <F, T> & __v) {

  if (__v.invalid()) {
    pack((unsigned)0);
  }
  else {
    pack((unsigned)1); 
    pack (__v.fitness ());
  }

  unsigned len = __v.size ();
  pack (len);
  for (unsigned i = 0 ; i < len; i ++)
    pack (__v [i]);  
}

template <class F, class T> void unpack (eoVector <F, T> & __v) {

  unsigned valid; unpack(valid);

  if (! valid) {
    __v.invalidate();
  }
  else {
    F fit; 
    unpack (fit);
    __v.fitness (fit);
  }

  unsigned len;
  unpack (len);
  __v.resize (len);
  for (unsigned i = 0 ; i < len; i ++)
    unpack (__v [i]);
}

template <class F, class T, class V> void pack (const eoVectorParticle <F, T, V> & __v) {

  if (__v.invalid()) {
    pack((unsigned)0);
  }
  else {
    pack((unsigned)1); 
    pack (__v.fitness ());
    pack (__v.best());
  }

  unsigned len = __v.size ();
  pack (len);
  for (unsigned i = 0 ; i < len; i ++)
    pack (__v [i]);  
  for (unsigned i = 0 ; i < len; i ++)
    pack (__v.bestPositions[i]); 
  for (unsigned i = 0 ; i < len; i ++)
    pack (__v.velocities[i]);  
}

template <class F, class T, class V> void unpack (eoVectorParticle <F, T, V> & __v) {

unsigned valid; unpack(valid);

  if (! valid) {
    __v.invalidate();
  }
  else {
    F fit; 
    unpack (fit);
    __v.fitness (fit);
      unpack(fit);
  __v.best(fit);
    
  }
  unsigned len;
  unpack (len);
  __v.resize (len);
  for (unsigned i = 0 ; i < len; i ++)
    unpack (__v [i]);  
  for (unsigned i = 0 ; i < len; i ++)
    unpack (__v.bestPositions[i]); 
  for (unsigned i = 0 ; i < len; i ++)
    unpack (__v.velocities[i]);  
}

template <class F, class T, class V, class W> void unpack (moeoVector <F,T,V,W> &_v) 
{
	unsigned valid;
	unpack(valid);
    if (! valid)
    	_v.invalidate();
    else 
    {
  		T fit; 
    	unpack (fit);
    	_v.fitness (fit);
  	}
  	unpack(valid);
    if (! valid)
    	_v.invalidateDiversity();
    else 
    {
  		V diver;
  		unpack(diver);
  		_v.diversity(diver);
    }
  	unsigned len;
    unpack (len);
    _v.resize (len);
    for (unsigned i = 0 ; i < len; i ++)
    	unpack (_v [i]);
    unpack(valid);
    if (! valid)
    	_v.invalidateObjectiveVector();
    else 
    {
    	F object;
    	unpack (len);
    	object.resize(len);
    	for (unsigned i = 0 ; i < len; i ++)
    		unpack (object[i]);
    	_v.objectiveVector(object);
    }
}


template <class F, class T, class V, class W> void pack (moeoVector <F,T,V,W> &_v) 
{
	if (_v.invalid())
    	pack((unsigned)0);
    else
    {
    	pack((unsigned)1); 
    	pack (_v.fitness ());
    }
    if (_v.invalidDiversity())
    	pack((unsigned)0);
    else
    {
    	pack((unsigned)1);
    	pack(_v.diversity());
    }
    unsigned len = _v.size ();
    pack (len);
    for (unsigned i = 0 ; i < len; i ++)
    	pack (_v[i]);
    if (_v.invalidObjectiveVector())
    	pack((unsigned)0);
    else
    {
    	pack((unsigned)1); 
    	F object;
    	object=_v.objectiveVector();
    	len=object.nObjectives();
    	pack (len);
    	for (unsigned i = 0 ; i < len; i ++)
    		pack (object[i]);	
    }
}

#endif
