/*
<customStochasticTopology.h>
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

#ifndef CUST_STOCH_TOPO_H_
#define CUST_STOCH_TOPO_H_

#include <vector>
#include <topology/abstractTopology.h>
#include <string>

namespace paradiseo
{
namespace smp
{

class CustomStochasticTopology : public AbstractTopology
{
public :
	
    /**
    *Constructor for a custom topology from a file
    *@param filename Name of the file, must be in the same directory as the executable
    */
	CustomStochasticTopology(std::string filename);
	
    /**
    * Inherited from AbstractTopology
    * @see smp::topology::AbstractTopology::getIdNeighbors
    */	
	std::vector<unsigned> getIdNeighbors(unsigned idNode) const;
	
	/**
	* Inherited from AbstractTopology : a custom topology cannot be reconstructed in a different way, so the method throws an error if nbNode doesn't match with the actual number of nodes.
	* @param nbNode number of nodes for the topology
	*/
	void construct(unsigned nbNode);
	
    /**
    *Inherited from AbstractTopology : changes the topology : removes any connection from/to the given node.
    *@param idNode index of the node to be isolated
    */
    void isolateNode(unsigned idNode);
	
private :
	std::vector<std::vector<double>> _matrix;
};

}

}

#endif
