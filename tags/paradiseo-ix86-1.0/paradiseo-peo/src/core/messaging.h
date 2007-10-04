/* 
* <messaging.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Alexandru-Adrian Tantar
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

#ifndef __mess_h
#define __mess_h

#include <utility>

/* Char */
extern void pack (const char & __c); 

/* Float */
extern void pack (const float & __f, int __nitem = 1); 

/* Double */
extern void pack (const double & __d, int __nitem = 1); 

/* Integer */
extern void pack (const int & __i, int __nitem = 1); 

/* Unsigned int. */
extern void pack (const unsigned int & __ui, int __nitem = 1); 

/* Short int. */
extern void pack (const short & __sh, int __nitem = 1); 

/* Unsigned short */
extern void pack (const unsigned short & __ush, int __nitem = 1);

/* Long */
extern void pack (const long & __l, int __nitem = 1); 

/* Unsigned long */
extern void pack (const unsigned long & __ul, int __nitem = 1); 

/* String */
extern void pack (const char * __str); 

/* Pointer */
template <class T> void pack (const T * __ptr) {
  
  pack ((unsigned long) __ptr); 
}

/* Pair */
template <class U, class V> void pack (const std :: pair <U, V> & __pair) {
  
  pack (__pair.first);
  pack (__pair.second);
}

//

/* Float */
extern void unpack (char & __c); 

/* Float */
extern void unpack (float & __f, int __nitem = 1); 

/* Double */
extern void unpack (double & __d, int __nitem = 1); 

/* Integer */
extern void unpack (int & __i, int __nitem = 1); 

/* Unsigned int. */
extern void unpack (unsigned int & __ui, int __nitem = 1); 

/* Short int. */
extern void unpack (short & __sh, int __nitem = 1); 

/* Unsigned short */
extern void unpack (unsigned short & __ush, int __nitem = 1);

/* Long */
extern void unpack (long & __l, int __nitem = 1); 

/* Unsigned long */
extern void unpack (unsigned long & __ul, int __nitem = 1); 

/* String */
extern void unpack (char * __str); 

/* Pointer */
template <class T> void unpack (T * & __ptr) {
  
  unsigned long p;
  unpack (p);
  __ptr = (T *) p;
}

/* Pair */
template <class U, class V> void unpack (std :: pair <U, V> & __pair) {
  
  unpack (__pair.first);
  unpack (__pair.second);
}

#endif
