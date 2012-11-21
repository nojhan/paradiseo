/*
<algoDispatching.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef ALGO_D_H_
#define ALGO_D_H_

#include <eo>

namespace paradiseo
{
namespace smp
{
/** Algo Dispatching

The Algo Dispatching enable to choose the algorithm to call at compile time by tag-dispatching method

**/

// Main algorithms tags
struct error_tag {};
struct eoEasyEA_tag {};
struct eoEasyPSO_tag {};
struct eoSyncEasyPSO_tag {};
 
template<template <class> class A, class B>
struct traits {
    typedef error_tag type;
};
        
template<class B>
struct traits<eoEasyEA,B> {
    typedef eoEasyEA_tag type;
};

template<class B>
struct traits<eoEasyPSO,B> {
    typedef eoEasyPSO_tag type;
};

template<class B>
struct traits<eoSyncEasyPSO,B> {
    typedef eoSyncEasyPSO_tag type;
};

}

}

#endif
