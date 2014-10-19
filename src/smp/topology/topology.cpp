/*
<topology.cpp>
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

template <class TopologyType>	
std::vector<unsigned> paradiseo::smp::Topology<TopologyType>::getIdNeighbors(unsigned idNode) const
{
	std::vector<unsigned> neighbors;
	for(unsigned j=0; j<_matrix.size();j++)
		if(_matrix[idNode][j]) neighbors.push_back(j);
		
	return neighbors;
}

template <class TopologyType>
void paradiseo::smp::Topology<TopologyType>::construct(unsigned nbNode)
{
    _builder(nbNode, _matrix);
}


template <class TopologyType>
void paradiseo::smp::Topology<TopologyType>::isolateNode(unsigned idNode)
{
    for(unsigned i = 0; i < _matrix.size(); i++)
    {
        //Line of idNode to false : no connection FROM this node
        _matrix[idNode][i] = false;         
        
        //Column of idNode to false : no connection TO this node
        _matrix[i][idNode] = false;
    }
}

template <class TopologyType>
TopologyType & paradiseo::smp::Topology<TopologyType>::getBuilder()
{
    TopologyType &b=_builder;
    return b;
}

