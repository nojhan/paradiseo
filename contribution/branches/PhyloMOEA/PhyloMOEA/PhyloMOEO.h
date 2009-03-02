/***************************************************************************
 *   Copyright (C) 2008 by Waldo Cancino   *
 *   wcancino@icmc.usp.br   *
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

#ifndef PHYLOMOEO_H_
#define PHYLOMOEO_H_

#include <moeo>
#include "phylotreeIND.h"
#include <iostream>

class ObjectiveVectorTraits : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int i)
    {
        return true;
    }
    static bool maximizing (int i)
    {
        return false;
    }
    static unsigned int nObjectives ()
    {
        return 2;
    }
};

typedef moeoRealObjectiveVector < ObjectiveVectorTraits > ObjectiveVector;

class PhyloMOEO : public  MOEO< ObjectiveVector, double, double >
{
private:
	phylotreeIND *tree;
	void copy(const PhyloMOEO &other)
	 { 
			if(tree!=NULL)delete tree;
			tree=NULL;
			if(other.tree!=NULL)tree = new phylotreeIND(other.get_tree());
			if(!other.invalidObjectiveVector())
				this->objectiveVector( other.objectiveVector() );
			else this->invalidateObjectiveVector();

			if(!other.invalidFitness())this->fitness( other.fitness() );
			else this->invalidateFitness();

			if(!other.invalidDiversity())this->diversity( other.diversity() );
			else this->invalidateDiversity();
	}
		
public:
    PhyloMOEO() : MOEO < ObjectiveVector, double, double >() { tree = NULL; }
	PhyloMOEO(const PhyloMOEO &other) : MOEO < ObjectiveVector, double, double >(other) { 
		tree=NULL;
		if(other.tree!=NULL)tree = new phylotreeIND(other.get_tree());
    }
	void set_tree_template(phylotreeIND &other) { if(tree!=NULL)delete tree; tree = other.clone(); };
	void set_random_tree(phylotreeIND &other) { if(tree!=NULL)delete tree; tree = other.randomClone(); };
	PhyloMOEO& operator= (const PhyloMOEO& other) {
			copy(other); 
			return *this; 
	}
	void   readFrom (std::istream &_is)
	{
		string s;
		_is >> s;
		tree->read_newick2( s );
	}
	phylotreeIND & get_tree() const { return *tree; }
	~PhyloMOEO() { if(tree!=NULL)delete tree; }
};
#endif


