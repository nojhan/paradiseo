/*
<customBooleanTopology.cpp>
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


#include <topology/customBooleanTopology.h>
#include <vector>
#include <assert.h>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <eo>

paradiseo::smp::CustomBooleanTopology::CustomBooleanTopology(std::string filename)
{
    std::ifstream f(filename);
    
    if(f)
    {
        int temp;
        unsigned size;
        bool isNeighbor, isFirst = true;
        std::string line;
        std::vector<bool> lineVector;
        
        while(getline(f, line))
        {
            lineVector.clear();
            
            //line contains a line of text from the file
            std::istringstream tokenizer(line);
            std::string token;
            
            while(tokenizer >> temp >> std::skipws)
            {
                //white spaces are skipped, and the integer is converted to boolean, to be stored
                isNeighbor = (bool) temp;
                lineVector.push_back(isNeighbor);
            }
            
            //if this is the first line, we must initiate the variable size
            if(isFirst)
            {
                size = lineVector.size();
                isFirst = false;
            }
            
            //for each vector non empty, if the size is not equal to the others, error
            if(lineVector.size() != size && !lineVector.empty())
                throw eoException("Mistake in the topology, line " + std::to_string(_matrix.size()+1) );
                
            if(!lineVector.empty())
                _matrix.push_back(lineVector);
        }

        //for each vector, verify if the size is equal to the size of the final matrix
        for(auto& line : _matrix)
            if(line.size() != _matrix.size())
                throw eoException("Mistake in the topology, matrix is not squared" );

        f.close () ;
    }   
    else
    {
        throw eoFileError(filename);
    }    
}

std::vector<unsigned> paradiseo::smp::CustomBooleanTopology::getIdNeighbors(unsigned idNode) const
{
	std::vector<unsigned> neighbors;
	for(unsigned j = 0; j < _matrix.size();j++)
		if(_matrix[idNode][j])
		    neighbors.push_back(j);
	
	return neighbors;
}

void paradiseo::smp::CustomBooleanTopology::construct(unsigned nbNode)
{
    assert(nbNode == _matrix.size());
}

void paradiseo::smp::CustomBooleanTopology::isolateNode(unsigned idNode)
{
    for(unsigned i = 0; i < _matrix.size(); i++)
    {
        //Line of idNode to false : no connection FROM this node
        _matrix[idNode][i] = false;         
        
        //Column of idNode to false : no connection TO this node
        _matrix[i][idNode] = false;
    }
}
