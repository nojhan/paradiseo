/*
  <moValueStat.h>
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

#ifndef moValueStat_h
#define moValueStat_h

#include <continuator/moStat.h>
#include <utils/eoParam.h>

/**
 * The statistic gives the number of iteration
 */
template <class EOT, class ValueType>
class moValueStat : public moStat<EOT, double>
{
public :
    using moStat<EOT, double>::value;

    /**
     * Default Constructor
     *
     * @param _valueParam the value to report in the statistic
     * @param _restart if true, the value is initialized at 0 for each init (restart)
     */
  moValueStat(eoValueParam<ValueType> & _valueParam, bool _restart = false): moStat<EOT, double>(0, "value"), valueParam(_valueParam), restart(_restart) {
  }

    /**
     * Init the number of iteration
     * @param _sol a solution
     */
    virtual void init(EOT & _sol) {
      if (restart)
        value_start = valueParam.value();
      else
	value_start = 0;

      value() = (double) (valueParam.value() - value_start);
    }

    /**
     * Set the number of iteration
     * @param _sol a solution
     */
    virtual void operator()(EOT & _sol) {
      value() = (double) (valueParam.value() - value_start);
    }

    /**
     * @return name of the class
     */
    virtual std::string className(void) const {
        return "moValueStat";
    }

private:
  eoValueParam<ValueType> & valueParam;

  bool restart;
  ValueType value_start ;

};

#endif
