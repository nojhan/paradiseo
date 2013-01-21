/*
<bimap.cpp>
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

template<class A, class B>
void paradiseo::smp::Bimap<A,B>::add(A& a, B& b)
{
    rightAssociation[a] = b;
    leftAssociation[b] = a;
}

template<class A, class B>
std::map<A,B> paradiseo::smp::Bimap<A,B>::getRight() const
{
    return rightAssociation;
}

template<class A, class B>
std::map<B,A> paradiseo::smp::Bimap<A,B>::getLeft() const
{
    return leftAssociation;
}

template<class A, class B>    
void paradiseo::smp::Bimap<A,B>::removeFromRight(const A& a)
{
    for(auto& it : leftAssociation)
        if(it->second == a)
            leftAssociation.erase(it);
        
    rightAssociation.erase(a);
}

template<class A, class B>    
void paradiseo::smp::Bimap<A,B>::removeFromLeft(const B& b) 
{
    for(auto& it : rightAssociation)
        if(it->second == b)
            rightAssociation.erase(it);
       
    leftAssociation.erase(b);
}

template<class A, class B>    
unsigned paradiseo::smp::Bimap<A,B>::size() const
{
    return leftAssociation.size();
}   

template<class A, class B>  
bool paradiseo::smp::Bimap<A,B>::empty() const
{
    return leftAssociation.empty();
}
