/*
  <moDummyExplorer.h>
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

#ifndef _moDummyExplorer_h
#define _moDummyExplorer_h

#include <explorer/moNeighborhoodExplorer.h>

/**
 * Dummy Explorer the neighborhood: nothing is explored
 */
template< class Neighbor >
class moDummyExplorer : public moNeighborhoodExplorer<Neighbor>
{
public:
    typedef moNeighborhood<Neighbor> Neighborhood;
    typedef typename Neighbor::EOT EOT;
    typedef typename EOT::Fitness Fitness ;

    moDummyExplorer(): moNeighborhoodExplorer<Neighbor>() { }

    /**
     * NOTHING TO DO
     * @param _solution unused solution
     */
    void initParam (EOT& /*_solution*/) { } ;

    /**
     * NOTHING TO DO
     * @param _solution unused solution
     */
    void updateParam (EOT& /*_solution*/) { } ;

    /**
     * NOTHING TO DO
     * @param _solution unused solution
     * @return always false
     */
    bool isContinue(EOT& /*_solution*/) {
        return false;
    } ;

    /**
     * NOTHING TO DO
     * @param _solution unused solution
     */
    void move(EOT& /*_solution*/) { } ;

    /**
     * NOTHING TO DO
     * @param _solution unused solution
     * @return always false
     */
    virtual bool accept(EOT& /*_solution*/) {
        return false;
    } ;

    /**
     * NOTHING TO DO
     * @param _solution unused solution
     */
    virtual void terminate(EOT& /*_solution*/) { } ;

    /**
     * NOTHING TO DO
     * @param _solution unused solution
     */
    void operator()(EOT & /*_solution*/) { }

    /**
     * Return the class name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moDummyExplorer";
    }

};

#endif
