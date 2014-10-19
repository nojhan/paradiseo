/*
* <FlowShopOpCrossoverQuad.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
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
//-----------------------------------------------------------------------------

#include "FlowShopOpCrossoverQuad.h"


std::string FlowShopOpCrossoverQuad::className() const
  {
    return "FlowShopOpCrossoverQuad";
  }


bool FlowShopOpCrossoverQuad::operator()(FlowShop & _flowshop1, FlowShop & _flowshop2)
{
  bool oneAtLeastIsModified;
  // computation of the 2 random points
  unsigned int point1, point2;
  do
    {
      point1 =  rng.random(std::min(_flowshop1.size(), _flowshop2.size()));
      point2 =  rng.random(std::min(_flowshop1.size(), _flowshop2.size()));
    }
  while (fabs((double) point1-point2) <= 2);
  // computation of the offspring
  FlowShop offspring1 = generateOffspring(_flowshop1, _flowshop2, point1, point2);
  FlowShop offspring2 = generateOffspring(_flowshop2, _flowshop1, point1, point2);
  // does at least one genotype has been modified ?
  if ((_flowshop1 != offspring1) || (_flowshop2 != offspring2))
    {
      // update
      _flowshop1.value(offspring1);
      _flowshop2.value(offspring2);
      // at least one genotype has been modified
      oneAtLeastIsModified = true;
    }
  else
    {
      // no genotype has been modified
      oneAtLeastIsModified = false;
    }
  // return 'true' if at least one genotype has been modified
  return oneAtLeastIsModified;
}


FlowShop FlowShopOpCrossoverQuad::generateOffspring(const FlowShop & _parent1, const FlowShop & _parent2, unsigned int _point1, unsigned int _point2)
{
  FlowShop result = _parent1;
  std::vector<bool> taken_values(result.size(), false);
  if (_point1 > _point2)
    std::swap(_point1, _point2);
  /* first parent */
  for (unsigned int i=0 ; i<=_point1 ; i++)
    {
      // result[i] == _parent1[i]
      taken_values[_parent1[i]] = true;
    }
  for (unsigned int i=_point2 ; i<result.size() ; i++)
    {
      // result[i] == _parent1[i]
      taken_values[_parent1[i]] = true;
    }
  /* second parent */
  unsigned int i = _point1+1;
  unsigned int j = 0;
  while (i<_point2 && j<_parent2.size())
    {
      if (! taken_values[_parent2[j]])
        {
          result[i] = _parent2[j];
          i++;
        }
      j++;
    }
  return result;
}
