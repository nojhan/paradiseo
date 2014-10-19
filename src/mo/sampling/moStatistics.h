/*
  <moStatistics.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

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

#ifndef moStatistics_h
#define moStatistics_h

#include <vector>
#include <cmath>
#include "../../eo/utils/eoDistance.h"

/**
 * Tools to compute some basic statistics
 *
 *   Remember it is better to use some statistic tool like R, etc.
 *   But it could be usefull to have here in paradisEO
 */
class moStatistics
{
public:
    /**
     * Default Constructor
     */
    moStatistics() { }

    /**
     * To compute min, max , average and standard deviation of a vector of double
     *
     * @param data vector of double
     * @param min to compute
     * @param max to compute
     * @param avg average to compute
     * @param std standard deviation to compute
     */
    void basic(const std::vector<double> & data,
               double & min, double & max, double & avg, double & std) {

        if (data.size() == 0) {
            min = 0.0;
            max = 0.0;
            avg = 0.0;
            std = 0.0;
        } else {
            unsigned int n = data.size();

            min = data[0];
            max = data[0];
            avg = 0.0;
            std = 0.0;

            double d;
            for (unsigned int i = 0; i < n; i++) {
                d = data[i];
                if (d < min)
                    min = d;
                if (max < d)
                    max = d;
                avg += d;
                std += d * d;
            }

            avg /= n;

            std = (std / n) - avg * avg ;
            if (std > 0)
                std = sqrt(std);
        }
    }

    /**
     * To compute the distance between solutions
     *
     * @param data vector of solutions
     * @param distance method to compute the distance
     * @param matrix matrix of distances between solutions
     */
    template <class EOT>
    void distances(const std::vector<EOT> & data, eoDistance<EOT> & distance,
                   std::vector< std::vector<double> > & matrix) {
        if (data.size() == 0) {
            matrix.resize(0);
        } else {
            unsigned int n = data.size();

            matrix.resize(n);
            for (unsigned i = 0; i < n; i++)
                matrix[i].resize(n);

            unsigned j;
            for (unsigned i = 0; i < n; i++) {
                matrix[i][i] = 0.0;
                for (j = 0; j < i; j++) {
                    matrix[i][j] = distance(data[i], data[j]);
                    matrix[j][i] = matrix[i][j];
                }
            }
        }
    }

    /**
     * To compute the autocorrelation and partial autocorrelation
     *
     * @param data vector of double
     * @param nbS number of correlation coefficients
     * @param rho autocorrelation coefficients
     * @param phi partial autocorrelation coefficients
     */
    void autocorrelation(const std::vector<double> & data, unsigned int nbS,
                         std::vector<double> & rho, std::vector<double> & phi) {
        if (data.size() == 0) {
            rho.resize(0);
            phi.resize(0);
        } else {
            unsigned int n = data.size();

            std::vector<double> cov;
            cov.resize(nbS+1);
            //double cov[nbS+1];
            std::vector<double> m;
            m.resize(nbS+1);
            //double m[nbS+1];
            std::vector<double> sig;
            sig.resize(nbS+1);
            //double sig[nbS+1];

            rho.resize(nbS+1);
            phi.resize(nbS+1);
            rho[0] = 1.0;
            phi[0] = 1.0;

            unsigned s, k;

            for (s = 0; s <= nbS; s++) {
                cov[s] = 0;
                m[s]   = 0;
                sig[s] = 0;
            }

            double m0, s0;
            unsigned j;

            k = 0;
            s = nbS;
            while (s > 0) {
                while (k + s < n) {
                    for (j = 0; j <= s; j++) {
                        m[j]   += data[k+j];
                        sig[j] += data[k+j] * data[k+j];
                        cov[j] += data[k] * data[k+j];
                    }
                    k++;
                }

                m[s]  /= n - s;
                sig[s] = sig[s] / (n - s) - m[s] * m[s];
                if (sig[s] <= 0)
                    sig[s] = 0;
                else
                    sig[s] = sqrt(sig[s]);
                m0 = m[0] / (n - s);
                s0 = sqrt(sig[0] / (n - s) - m0 * m0);
                cov[s] = cov[s] / (n - s) - (m[0] / (n - s)) * m[s];
                rho[s] = cov[s] / (sig[s] * s0);
                s--;
            }

            std::vector< std::vector<double> > phi2;
            phi2.resize(nbS+1);
            for(unsigned int i=0; i<phi2.size(); i++)
            	phi2[i].resize(nbS+1);
            //double phi2[nbS+1][nbS+1];
            double tmp1, tmp2;

            phi2[1][1] = rho[1];
            for (k = 2; k <= nbS; k++) {
                tmp1 = 0;
                tmp2 = 0;
                for (j = 1; j < k; j++) {
                    tmp1 += phi2[k-1][j] * rho[k-j];
                    tmp2 += phi2[k-1][j] * rho[j];
                }
                phi2[k][k] = (rho[k] - tmp1) / (1 - tmp2);
                for (j = 1; j < k; j++)
                    phi2[k][j] = phi2[k-1][j] - phi2[k][k] * phi2[k-1][k-j];
            }

            for (j = 1; j <= nbS; j++)
                phi[j] = phi2[j][j];

        }
    }

};


#endif
