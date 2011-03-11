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

#include "SubsModel.h"
//#include <iostream>

using namespace std;

SubstModel::SubstModel(Sequences &p, int m)
{
	model = m;
	patterns = &p;
	eigensystem = NULL;
	patterns->calculate_frequences();
	frequences = patterns->frequences();
	kappa = 4;
	//kappa = 32.451;
	a=b=c=d=e=1;
	ievec = new double*[4];
	for(int i=0; i<4; i++)ievec[i] = new double[4];
}

void SubstModel::init()
{
	init_rate_matrix();
	switch(model)
	{
		case JC69:
			jc69();break;
		case F81:
			f81();break;
		case K2P:
			k2p();break;
		case HKY85:
			hky85();break;
		case GTR:
			gtr();break;
		default:
		{
			model = 1;
			f81(); break;
		}
	}
	// calculate eigenvalues and eigenvectors
	if(eigensystem!=NULL)delete eigensystem;
	eigensystem = new EigenSolver(rate, 4);
	eigensystem->solve();
	// calculat inverse eigenvectors
	luinverse(eigensystem->eigenvectors(), ievec, NUM_AA);
}

void SubstModel::init_rate_matrix()
{
	rate[0] = rate[1] = rate[2] = rate[3] = 
	rate[4] = rate[5] = rate[6] = rate[7] = 
	rate[8] = rate[9] = rate[10] = rate[11] = 
	rate[12] = rate[13] = rate[14] = rate[15] = 1.0;
}


void SubstModel::construct_rate_matrix()
{
	// multiply the off-diagonal elements by the frequences
	mult_frequences();
	// calculate digonal elements
	set_diagonal();
	// normalize the rate matrix
	normalize();
}


// multiple the off-diagonal elementes by the frequences

void SubstModel::mult_frequences()
{
	for (int i = 0; i < 4; i++)  {
		for (int j = i+1; j < 4; j++) {
			set_rate(i,j, get_rate(i,j) * frequences[j]);
			set_rate(j,i, get_rate(j,i) * frequences[i]);
		}
	}
}


void SubstModel::set_diagonal()
{
// set the diagonal elements
	for(int i=0; i<4; i++)
	{
		double sum = 0;
		for(int j=0; j<4; j++) if(i!=j)sum+=get_rate(i,j);
		set_rate(i,i, -sum);
	}
}

// normaliza the rate matrix
void SubstModel::normalize()
{
	double tmp = 0.0;

	for (int i = 0; i < 4; i++)
	{
		tmp += -get_rate(i,i)*frequences[i];
	}
	//std::cout << "Rate matrix:" << std::endl;
	//std::cout << "----------------------------------" << std::endl;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			set_rate(i,j, get_rate(i,j)/tmp);
			//std::cout << get_rate(i,j) << "  |  ";
		}
		//std::cout << std::endl;
	}
}


void SubstModel::print_rate_matrix()
{
	std::cout << "Rate matrix:" << std::endl;
	std::cout << "----------------------------------" << std::endl;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			std::cout << get_rate(i,j) << "  |  ";
		}
		std::cout << std::endl;
	}
}


// useful for models that don't complain about nucleotide frequences
void SubstModel::set_equal_frequences()
{
	frequences[0] = frequences[1] = frequences[2] = frequences[3] = 0.25;
}

// initialize the variables and matrizes


void SubstModel::f81()
{
	// set the matrix
	// parameter mu = 1.0
	// nothing to do, only construct the matrix
	construct_rate_matrix();
}


void SubstModel::jc69()
{
	// set equal frequences
	set_equal_frequences();
	construct_rate_matrix();
}

void SubstModel::hky85()
{
	//cout << "criando hky85 com kappa:" << kappa << endl;
	// transition/traversion ratio
	set_rate(0,2, kappa);
	set_rate(1,3, kappa);
	set_rate(2,0, kappa);
	set_rate(3,1, kappa);
	// multfrequences
	construct_rate_matrix();
	
}

void SubstModel::k2p()
{
	// set equal frequences
	set_equal_frequences();
 	// the same of HKY85 except by the equal frequences
	hky85();
}

void SubstModel::gtr()
{

	a=1.23767538;	b=3.58902963;
	c=2.16811705;	d=0.73102339;	e=6.91039679;
	set_rate(0,1, a);
	set_rate(1,0, a);
	set_rate(0,2, b);
	set_rate(2,0, b);
	set_rate(0,3, c);
	set_rate(3,0, c);
	set_rate(1,2, d);
	set_rate(2,1, d);
	set_rate(1,3, e);
	set_rate(3,1, e);
	// multfrequences
	construct_rate_matrix();
}
