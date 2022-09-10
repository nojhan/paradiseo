/*
<moIterContinuator.h>
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

#ifndef _moIterContinuator_h
#define _moIterContinuator_h

#include <continuator/moContinuator.h>
#include <neighborhood/moNeighborhood.h>

/**
 * Continue until a maximum fixed number of iterations is reached
 */
template< class Neighbor >
class moIterContinuator : public moContinuator<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;

    /**
     * @param _maxIter number maximum of iterations
     * @param _verbose true/false : verbose mode on/off
     */
    moIterContinuator(unsigned int _maxIter, bool _verbose=true): maxIter(_maxIter), verbose(_verbose) {}

    /**
     *@param _solution a solution
     *@return true if counter < maxIter
     */
    virtual bool operator()(EOT & /*_solution*/) {
        bool res;
        cpt++;
        res = (cpt < maxIter);
        if (!res && verbose)
            std::cout << "STOP in moIterContinuator: Reached maximum number of iterations [" << cpt << "/" << maxIter << "]" << std::endl;
        return res;
    }

    /**
     * reset the counter of iteration
     * @param _solution a solution
     */
    virtual void init(EOT & /*_solution*/) {
      cpt = 0;
    }

    /**
     * the current number of iteration
     * @return the number of iteration
     */
    unsigned int value() {
        return cpt ;
    }

private:
    unsigned int maxIter;
    unsigned int cpt;
    bool verbose;

};
#endif
