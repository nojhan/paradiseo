/*
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * (c) Antonio LaTorre <atorre@fi.upm.es>, 2007
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 * ParadisEO WebSite : http://paradiseo.gforge.inria.fr
 * Contact: paradiseo-help@lists.gforge.inria.fr
 *
 */

#ifndef _eoVRPEvalFunc_h
#define _eoVRPEvalFunc_h

// General includes
#include <stdexcept>
#include <fstream>

// The base definition of eoEvalFunc
#include "eoEvalFunc.h"

// Utilities for the VRP-TW problem
#include "eoVRPUtils.h"

/**
  * \class eoVRPEvalFunc eoVRPEvalFunc.h
  * \brief Evaluates an individual of type eoVRP.
  */

class eoVRPEvalFunc : public eoEvalFunc<eoVRP> {

public:

    /**
      * \brief Constructor: nothing to do here.
      */

    eoVRPEvalFunc () {

    }


    /**
      * \brief Computes the (penalized) fitness
      * \param _eo The individual to be evaluated.
      */

    void operator () (eoVRP& _eo) {

        double fit = 0.0;

        if (_eo.decoded ()) {

            if (_eo.invalid ()) {

                std::cerr << "Warning: invalid individual presents decoding information." << std::endl;
                fit = _eo.decode ();

            }
            else
                fit = _eo.length ();

        }
        else {

            if (!_eo.invalid ()) {

                std::cerr << "Warning: valid individual does not present decoding information." << std::endl;
                std::cerr << "         Proceeding to decode..." << std::endl;

            }

            fit = _eo.decode ();

        }

        _eo.fitness (fit);

    }


private:


};

#endif
