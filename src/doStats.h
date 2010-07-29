/*
 *             Copyright (C) 2005 Maarten Keijzer
 *
 *          This program is free software; you can redistribute it and/or modify
 *          it under the terms of version 2 of the GNU General Public License as
 *          published by the Free Software Foundation.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program; if not, write to the Free Software
 *          Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef _doStats_h
#define _doStats_h

#include <eoPrintable.h>

class doStats : public eoPrintable
{
public:
    doStats();

    virtual void printOn(std::ostream&) const;

protected:
    double _n;
};

class doMean : public doStats
{
public:
    doMean();

    virtual void update(double);
    virtual void printOn(std::ostream&) const;

    double get_mean() const;

protected:
    double _mean;
};

class doVar : public doMean
{
public:
    doVar();

    virtual void update(double);
    virtual void printOn(std::ostream&) const;

    double get_var() const;
    double get_std() const;

protected:
    double _sumvar;
};

/** Single covariance between two variates */
class doCov : public doStats
{
public:
    doCov();

    virtual void update(double, double);
    virtual void printOn(std::ostream&) const;

    double get_meana() const;
    double get_meanb() const;
    double get_cov()   const;

protected:
    double _meana;
    double _meanb;
    double _sumcov;
};

class doCovMatrix : public doStats
{
public:
    doCovMatrix(unsigned dim);

    virtual void update(const std::vector<double>&);

    double get_mean(int) const;
    double get_var(int) const;
    double get_std(int) const;
    double get_cov(int, int) const;

protected:
    std::vector< double > _mean;
    std::vector< std::vector< double > > _sumcov;
};

class doHyperVolume : public doStats
{
public:
    doHyperVolume();

    virtual void update(double);
    virtual void printOn(std::ostream&) const;

    double get_hypervolume() const;

protected:
    double _hv;
};

class doCholesky : public doStats
{
    
};

#endif // !_doStats_h
