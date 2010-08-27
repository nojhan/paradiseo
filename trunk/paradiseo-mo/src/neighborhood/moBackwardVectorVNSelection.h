/*
<moBackwardVectorVNSelection.h>
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

#ifndef _moBackwardVectorVNSelection_h
#define _moBackwardVectorVNSelection_h

#include <neighborhood/moVectorVNSelection.h>

template< class EOT >
class moBackwardVectorVNSelection: public moVectorVNSelection<EOT>{

	using moVectorVNSelection<EOT>::LSvector;
	using moVectorVNSelection<EOT>::shakeVector;
	using moVectorVNSelection<EOT>::current;

public:

	moBackwardVectorVNSelection(eoMonOp<EOT>& _firstLS, eoMonOp<EOT>& _firstShake, bool _cycle):moVectorVNSelection<EOT>(_firstLS, _firstShake), cycle(_cycle){}

    /**
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moBackwardVectorVNSelection";
    }

    /**
     * test if there is still some neighborhood to explore
     * @return true if there is some neighborhood to explore
     */
    virtual bool cont(EOT& _solution, eoMonOp<EOT>& _shake, eoMonOp<EOT>& _ls){
    	return (cycle || (current >0 ));
    }

    /**
     * put the current neighborhood on the first one
     */
    virtual void init(EOT& _solution, eoMonOp<EOT>& _shake, eoMonOp<EOT>& _ls){
    	current=LSvector.size()-1;
    	_ls=LSvector[current];
    	_shake = shakeVector[current];
    }

    /**
     * put the current neighborhood on the next one
     */
    virtual void next(EOT& _solution, eoMonOp<EOT>& _shake, eoMonOp<EOT>& _ls){
    	current= (current + LSvector.size() -1) % LSvector.size();
    	_ls=LSvector[current];
    	_shake = shakeVector[current];
    }

private:

	bool cycle;

};

#endif
