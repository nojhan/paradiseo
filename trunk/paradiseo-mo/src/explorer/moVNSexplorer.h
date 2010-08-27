/*
<moVNSexplorer.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef _moVNSexplorer_h
#define _moVNSexplorer_h

#include <explorer/moNeighborhoodExplorer.h>
#include <neighborhood/moVariableNeighborhoodSelection.h>
#include <eoOp.h>
#include <acceptCrit/moAcceptanceCriterion.h>

/**
 * Explorer for Variiable Neighborhood Search
 */
template< class Neighbor>
class moVNSexplorer : public moNeighborhoodExplorer< Neighbor >
{
public:

	typedef typename Neighbor::EOT EOT;

    /**
     * Constructor
     * @param _neighborhood the neighborhood
     * @param _eval the evaluation function
     * @param _neighborComparator a neighbor comparator
     * @param _solNeighborComparator solution vs neighbor comparator
     */
    moVNSexplorer(
    		moVariableNeighborhoodSelection<EOT> & _selection,
			moAcceptanceCriterion<Neighbor>& _acceptCrit):
    			moNeighborhoodExplorer<Neighbor>(), selection(_selection), shake(NULL), ls(NULL), acceptCrit(_acceptCrit), stop(false)
    {}

    /**
     * Destructor
     */
    ~moVNSexplorer() {
    }

    /**
     * initParam: NOTHING TO DO
     */
    virtual void initParam(EOT& _solution) {
    	selection.init(_solution, *shake, *ls);
    };

    /**
     * updateParam: NOTHING TO DO
     */
    virtual void updateParam(EOT & _solution) {
    	if ((*this).moveApplied()) {
    		selection.init(_solution, *shake, *ls);
    	}
    	else if (selection.cont(_solution, *shake, *ls)){
    		selection.next(_solution, *shake, *ls);
    	}
    	else
    		stop=true;
    };

    /**
     * terminate: NOTHING TO DO
     */
    virtual void terminate(EOT & _solution) {};

    /**
     * Explore the neighborhood of a solution
     * @param _solution
     */
    virtual void operator()(EOT & _solution) {
    	current=_solution;
    	(*shake)(current);
    	(*ls)(current);
    };

    /**
     * continue if a move is accepted
     * @param _solution the solution
     * @return true if an ameliorated neighbor was be found
     */
    virtual bool isContinue(EOT & _solution) {
    	return !stop;
    };

    /**
     * move the solution with the best neighbor
     * @param _solution the solution to move
     */
    virtual void move(EOT & _solution) {
    	_solution=current;
    };

    /**
     * accept test if an amelirated neighbor was be found
     * @param _solution the solution
     * @return true if the best neighbor ameliorate the fitness
     */
    virtual bool accept(EOT & _solution) {
    	return acceptCrit(_solution, current);
    };

    /**
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moVNSexplorer";
    }

private:
	moVariableNeighborhoodSelection<EOT>& selection;
	eoMonOp<EOT>* ls;
	eoMonOp<EOT>* shake;
	moAcceptanceCriterion<Neighbor>& acceptCrit;

	bool stop;
    EOT current;

};


#endif
