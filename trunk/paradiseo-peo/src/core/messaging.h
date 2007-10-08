// "messaging.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
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

/* Char */
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
