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

#ifndef PHYLOMOEO_EVAL_H_
#define PHYLOMOEO_EVAL_H_
#include <PhyloMOEO.h>
#include <parsimonycalculator.h>
#include <likelihoodcalculator.h>
//#include <peo>
#include <iostream>
#include <utils.h>

extern ProbMatrixContainer *probmatrixs_ptr;

class PhyloEval : public moeoEvalFunc < PhyloMOEO >
{
public:

	PhyloEval( ParsimonyCalculator &x, LikelihoodCalculator &y): parseval(x), likeval(y) { }
	
    void operator () (PhyloMOEO & _sol)
    {
		//cout << "hello im evaluating in the node " << getNodeRank() << endl;
		//if (_sol.invalidObjectiveVector())
        //{

        ObjectiveVector objVec;
		//if(! (_sol.get_tree().splits_valid() ) ) _sol.get_tree().calculate_splits();
		parseval.set_tree(_sol.get_tree());
		likeval.set_tree(_sol.get_tree());
		//objVec[0] = 1;
		//objVec[1] = 1;
        objVec[0] = parseval.fitch();
        objVec[1] = -likeval.calculate_likelihood();
        _sol.objectiveVector(objVec);
	probmatrixs_ptr->clear();
		//}
    }

private:
	LikelihoodCalculator &likeval;
	ParsimonyCalculator &parseval;
};


class PhyloLikelihoodTimeEval : public moeoEvalFunc < PhyloMOEO >
{
public:

	PhyloLikelihoodTimeEval( LikelihoodCalculator &y): likeval(y) { }
	
    void operator () (PhyloMOEO & _sol)
    {
        ObjectiveVector objVec;
		likeval.set_tree(_sol.get_tree());
		struct timeval start;
		gettimeofday(&start,NULL);

        objVec[0] = -likeval.calculate_likelihood();
		struct timeval end;
		gettimeofday(&end,NULL);

		print_elapsed_time_short(&start,&end);
		_sol.objectiveVector(objVec);
		cout << endl;
    }

private:
	LikelihoodCalculator &likeval;
};


#endif
