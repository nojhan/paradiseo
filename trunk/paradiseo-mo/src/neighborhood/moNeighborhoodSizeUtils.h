/*
  <moNeighborhoodSizeUtils.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Karima Boufaras, Thé Van LUONG

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

#ifndef _moNeighborhoodSizeUtils_h
#define _moNeighborhoodSizeUtils_h

/**
 * @return the factorial of an integer
 * @param _i an integer
 */

static int factorial(int _i) {
	if (_i == 0)
		return 1;
	else
		return _i * factorial(_i - 1);
}

/**
 * @return the neighborhood Size from the solution size and the exchange distance
 * @param _n the solution size
 * @param _xChange the number of positions to change
 */

static int sizeMapping(int _n, unsigned int _xChange) {
	int _sizeMapping;
	_sizeMapping = _n;
	for (int i = _xChange-1; i > 0; i--) {
		_sizeMapping *= (_n - i);
	}
	_sizeMapping /= factorial(_xChange);
	return _sizeMapping;
}


#endif
