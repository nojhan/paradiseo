/*
<t-moStatistics.cpp>
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

#include <iostream>
#include <cstdlib>
#include <cassert>

#include <paradiseo/mo/sampling/moStatistics.h>
#include "moTestClass.h"
#include <paradiseo/eo/utils/eoDistance.h>

int main() {

    std::cout << "[t-moStatistics] => START" << std::endl;

    moStatistics test;

    double min;
    double max;
    double avg;
    double std;

    //test des stats basic
    std::vector<double> sampling;
    sampling.push_back(3);
    sampling.push_back(5);
    sampling.push_back(2);
    sampling.push_back(4);


    test.basic(sampling, min, max, avg, std);
    assert(min==2);
    assert(max==5);
    assert(avg==3.5);
    //assert(std*std==1.25);


    sampling.resize(0);
    test.basic(sampling, min, max, avg, std);
    assert(min==0);
    assert(max==0);
    assert(avg==0);
    assert(std==0);

    //test de la distance
    std::vector<bitVector> data;
    eoHammingDistance<bitVector> dist;
    bitVector tmp(4,true);
    data.push_back(tmp);
    tmp[0]=false;
    data.push_back(tmp);
    tmp[2]=false;
    data.push_back(tmp);

    std::vector< std::vector<double> > matrix;

    test.distances(data, dist, matrix);

    assert(matrix[0][0]==0.0);
    assert(matrix[0][1]==1.0);
    assert(matrix[0][2]==2.0);

    assert(matrix[1][0]==1.0);
    assert(matrix[1][1]==0.0);
    assert(matrix[1][2]==1.0);

    assert(matrix[2][0]==2.0);
    assert(matrix[2][1]==1.0);
    assert(matrix[2][2]==0.0);

    //test de l'autocorrelation
    std::vector<double> rho, phi;
    test.autocorrelation(sampling, 2, rho, phi);

    sampling.push_back(3);
    sampling.push_back(5);
    sampling.push_back(2);
    sampling.push_back(4);

    test.autocorrelation(sampling, 2, rho, phi);

    std::cout << "[t-moStatistics] => OK" << std::endl;

    return EXIT_SUCCESS;
}

