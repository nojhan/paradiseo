/*
  <moMinNeighborStat.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.

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

#ifndef moMinNeighborStat_h
#define moMinNeighborStat_h

#include "moStat.h"
#include "moNeighborhoodStat.h"
#include "../neighborhood/moNeighborhood.h"

/**
 * From moNeighborhoodStat, to compute the min fitness in the neighborhood
 */
template< class Neighbor >
class moMinNeighborStat : public moStat<typename Neighbor::EOT, typename Neighbor::EOT::Fitness>
{
public :
    typedef typename Neighbor::EOT EOT ;
    typedef typename EOT::Fitness Fitness ;

    using moStat< EOT, Fitness >::value;

    /**
     * Constructor
     * @param _nhStat a neighborhoodStat
     */
    moMinNeighborStat(moNeighborhoodStat<Neighbor> & _nhStat):
            moStat<EOT, Fitness>(Fitness(), "min"), nhStat(_nhStat) {}

    /**
     * Set the worst fitness in the neighborhood
     * @param _sol the first solution
     */
    virtual void init(EOT & _sol) {
        value() = nhStat.getMin();
    }

    /**
     * Set the worst fitness in the neighborhood
     * @param _sol the corresponding solution
     */
    virtual void operator()(EOT & _sol) {
        value() = nhStat.getMin();
    }

    /**
     * @return the class name
     */
    virtual std::string className(void) const {
        return "moMinNeighborStat";
    }

private:
    /** moNeighborhoodStat */
    moNeighborhoodStat<Neighbor> & nhStat;
};

#endif
