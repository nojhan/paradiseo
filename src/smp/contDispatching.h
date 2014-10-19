/*
<contDispatching.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy, Thibault Lasnier - INSA Rouen

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

#ifndef SMP_CONT_DISPATCHING_H_
#define SMP_CONT_DISPATCHING_H_

/** Continuator Dispatching

The Continuator Dispatching enables to wrap continuators in the Island constructor for a better user interface and, moreover, avoiding side effect.

**/

template<class T, class U>
U& wrap_pp_impl(T&, U& u, std::false_type)
{ 
    return u; 
}

template<class T, class U>
T& wrap_pp_impl(T& t, U&, std::true_type)
{ 
    return t; 
}

template<class T, class U>
struct result_of_wrap_pp : 
    std::conditional<std::is_base_of<T,U>::value,T&,U&>
{};

template<class T, class U>
typename result_of_wrap_pp<T,U>::type wrap_pp(T& t, U& u)
{ 
    typedef typename std::is_base_of<T,U>::type tag;
    return wrap_pp_impl(t,u,tag());
}

#endif
