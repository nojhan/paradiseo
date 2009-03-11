/***************************************************************************
 *   Copyright (C) 2005 by Waldo Cancino                                   *
 *   wcancino@icmc.usp.br                                                  *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef PROBMATRIXCONTAINER_H
#define PROBMATRIXCONTAINER_H
#include "ProbMatrix.h"
#include <map>
/**
@author Waldo Cancino
*/
// container for an arbitrary number of Probmatrix under certain
// evolution model


class ProbMatrixContainer{

private:
	SubstModel *model;
	// contains the probamatirx
	std::map<double, ProbMatrix *> container;
public:
	ProbMatrix &operator[] (double branchlength)
	{
		if (container.find(branchlength)!=container.end() )
		{
			return *container[branchlength];
		}
		else
		{
			ProbMatrix *new_prob_matrix = new ProbMatrix(model, branchlength);
			new_prob_matrix->init();
			container[branchlength] = new_prob_matrix;
			return *new_prob_matrix;
		}
	}
	
	ProbMatrixContainer( SubstModel &m )
	{
		model = &m;
	}

	void change_matrix(double bl_old, double bl_new)
	{
		ProbMatrix &p = (*this)[bl_old];
		p.set_branch_length(bl_new);
		std::map<double, ProbMatrix *>::iterator it = container.find(bl_old);
		container.erase(it);
		it = container.find(bl_new);
		// prevent duplication
		if(it!=container.end()) delete (*it).second;
		/*{
			std::cout << " duplication " << bl_new << std::endl;
			container.erase(it);
		}*/
		// replace the container
		container[bl_new] = &p;
	}
	
	void clear()
	{
		std::map<double, ProbMatrix *>::iterator it = container.begin();
		std::map<double, ProbMatrix *>::iterator end = container.end();
		while(it!=end)
		{
			delete (*it).second;
			it++;
		}
		container.clear();
	}

	~ProbMatrixContainer() { 
		//std::cout << "limpando ..." << container.size() <<  " matrizes alocadas\n"; 
		clear(); }

};

#endif
