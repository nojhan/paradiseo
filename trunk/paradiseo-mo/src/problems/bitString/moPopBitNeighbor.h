/*
<moPopBitNeighbor.h>
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

#ifndef _moPopBitNeighbor_h
#define _moPopBitNeighbor_h

#include <eoPop.h>
#include <ga/eoBit.h>
#include <problems/bitString/moPopSol.h>
#include <neighborhood/moBackableNeighbor.h>
#include <neighborhood/moIndexNeighbor.h>

/**
 * Neighbor related to a vector of Bit
 */
template< class Fitness >
class moPopBitNeighbor : public moBackableNeighbor< moPopSol<eoBit<Fitness> > >, public moIndexNeighbor< moPopSol<eoBit<Fitness> > >
{
public:
    typedef moPopSol<eoBit<Fitness> > EOT ;

    using moBackableNeighbor<EOT>::fitness;
    using moIndexNeighbor<EOT>::key;

    /**
     * move the solution
     * @param _solution the solution to move
     */
    virtual void move(EOT & _solution) {
    	if(_solution.size()>0){
    		size=_solution[0].size();
    		_solution[key/size][key%size] = !_solution[key/size][key%size];
//    		fit=_solution[key/size].fitness();
    		_solution[key/size].invalidate();
//    		fitSol=_solution.fitness();
            _solution.invalidate();
    	}
    }

    /**
     * move back the solution (useful for the evaluation by modif)
     * @param _solution the solution to move back
     */
    virtual void moveBack(EOT & _solution) {
        _solution[key/size][key%size] = !_solution[key/size][key%size];
//        _solution[key/size].fitness(fit);
    }

    /**
     * return the class name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moPopBitNeighbor";
    }

    /**
     * Read object.\
     * Calls base class, just in case that one had something to do.
     * The read and print methods should be compatible and have the same format.
     * In principle, format is "plain": they just print a number
     * @param _is a std::istream.
     * @throw runtime_std::exception If a valid object can't be read.
     */
    virtual void readFrom(std::istream& _is) {
        std::string fitness_str;
        int pos = _is.tellg();
        _is >> fitness_str;
        if (fitness_str == "INVALID") {
            throw std::runtime_error("invalid fitness");
        }
        else {
            Fitness repFit ;
            _is.seekg(pos);
            _is >> repFit;
            _is >> key;
            fitness(repFit);
        }
    }

    /**
     * Write object. Called printOn since it prints the object _on_ a stream.
     * @param _os A std::ostream.
     */
    virtual void printOn(std::ostream& _os) const {
        _os << fitness() << ' ' << key << std::endl;
    }

private:
    unsigned int size;
};

#endif
