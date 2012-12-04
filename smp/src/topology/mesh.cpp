/*
<mesh.cpp>
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

#include <vector>
#include <cmath>
#include <topology/mesh.h>

void paradiseo::smp::Mesh::operator()(unsigned nbNode, std::vector<std::vector<bool>>& matrix) const
{
    int i=0, j, height, width;
    std::vector<unsigned> listFact = paradiseo::smp::Mesh::factorization(nbNode);
    int nbFact = listFact.size();
	    
	//Compute width and height
	//find the ratio height/width of the grid that matches best the variable _ratio
	while (i<listFact.size()-1 && (double)listFact[i]*listFact[i]/nbNode<_ratio)
	    i++;

    /*
    listFact[i] contains first factor which produces a ratio above the variable _ratio,
    or the last element if there is no ratio that can go over the variable _ratio. 
	*/
	double r1 = (double)listFact[i]*listFact[i]/nbNode;
	double r2 = (double)listFact[i-1]*listFact[i-1]/nbNode;
	
	//which ratio (r1,r2) matches _ratio best?
    if (std::abs(r2-_ratio) <= _ratio-r1)
        height = listFact[i-1];
    else
        height = listFact[i];
    
    width = nbNode/height;
	
	//Building matrix
	matrix.clear();
	
	matrix.resize(nbNode);
	for(auto& line : matrix)
	    line.resize(nbNode);
	    
	for(i = 0; i < matrix.size(); i++)
	{
	    matrix[i][i]=false;
	    for (j = i+1; j < matrix.size(); j++)
	    {
	        matrix[j][i] = matrix[i][j] = ((j-i == 1) && (j % width != 0)) || (j-i == width);
	    }
	}
}

void paradiseo::smp::Mesh::setRatio(double r)
{
    _ratio = r;
}

std::vector<unsigned> paradiseo::smp::Mesh::factorization(unsigned n) const
{
    unsigned i;
    double max = std::sqrt(n+1);
    std::vector<unsigned> listFact;
    for(i=1; i < max; i++)
    {
        if((n/i)*i == n)
            listFact.push_back(i);
    }
    return listFact;
}
