// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// gprop.h
// (c) GeNeura Team 1998
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */
//-----------------------------------------------------------------------------


#ifndef gprop_h
#define gprop_h

//-----------------------------------------------------------------------------

#include <iostream>               // istream ostream
#include <iomanip>                // setprecision
#include <string>                 // string
#include <EO.h>                   // EO
#include <eoOp.h>                 // eoMonOp eoQuadraticOp
#include <eoInit.h>               // eoInit
#include <utils/rnd_generators.h> // normal_generator
#include "mlp.h"                  // mlp::net mlp::set
#include "qp.h"                   // qp::set
#include "mse.h"                  // mse::error

//-----------------------------------------------------------------------------
// phenotype
//-----------------------------------------------------------------------------

struct phenotype
{
  int trn_ok, val_ok, tst_ok;
  double mse_error;

  static int trn_max, val_max, tst_max;

  friend bool operator<(const phenotype& a, const phenotype& b)
  {
      return (a.val_ok < b.val_ok) || ((!(b.val_ok < a.val_ok)) && (b.mse_error < a.mse_error));
  }

  friend bool operator==(const phenotype& a, const phenotype& b)
  {
      return (a.val_ok == b.val_ok) && (b.mse_error == a.mse_error);
  }

  friend bool operator>=(const phenotype& a, const phenotype& b)
  {
    return !(a < b);
  }

  friend bool operator>(const phenotype& a, const phenotype& b)
  {
    return (!(a == b)) && (!(a < b));
  }



  friend std::ostream& operator<<(std::ostream& os, const phenotype& p)
  {
    return os << p.trn_ok << "/" << p.trn_max << " "
	      << p.val_ok << "/" << p.val_max << " "
	      << p.tst_ok << "/" << p.tst_max << " "
	      << p.mse_error;
  }

  friend std::istream& operator>>(std::istream& is, phenotype& p)
  {
    return is; // complete me
  }
};


int phenotype::trn_max = 0, phenotype::val_max = 0, phenotype::tst_max = 0;

//-----------------------------------------------------------------------------
// genotype
//-----------------------------------------------------------------------------
#ifndef GPROP_GENOTYPE
#define GPROP_GENOTYPE mlp::net
#endif

typedef GPROP_GENOTYPE genotype;

//-----------------------------------------------------------------------------
// Chrom
//-----------------------------------------------------------------------------

extern unsigned in, out, hidden;

class Chrom: public EO<phenotype>, public genotype
{
public:
  Chrom(): genotype(in, out, std::vector<unsigned>(hidden < 1 ? 0 : 1, hidden)) {}

  std::string className() const { return "Chrom"; }

  void printOn (std::ostream& os) const
  {
    os << std::setprecision(3) << static_cast<genotype>(*this) << "  \t"
       << fitness();
    // os << fitness();
  }

  void readFrom (std::istream& is)
  {
    invalidate(); // complete me
  }
};

//-----------------------------------------------------------------------------
// eoChromInit
//-----------------------------------------------------------------------------

class eoInitChrom: public eoInit<Chrom>
{
public:
  void operator()(Chrom& chrom)
  {
    chrom.reset();
    chrom.invalidate();
  }
};

//-----------------------------------------------------------------------------
// global variables
//-----------------------------------------------------------------------------

mlp::set *trn_set = 0, *val_set = 0, *tst_set = 0;

void gprop_use_datasets(mlp::set *trn, mlp::set *val, mlp::set *tst) {
   trn_set = trn;
   val_set = val;
   tst_set = tst;
}

void ensure_datasets_initialized() {
    if (!trn_set) {
	std::cerr << "trn_set is not initialized.  Must call gprop_use_datasets before training\n";
	std::cerr.flush();
	abort();
    }
}

//-----------------------------------------------------------------------------
// eoChromMutation
//-----------------------------------------------------------------------------

class eoChromMutation: public eoMonOp<Chrom>
{
public:
  bool operator()(Chrom& chrom)
  {
    mse::net tmp(chrom);
    tmp.train(*trn_set, 10, 0, 0.001);
    return true;
  }
};

//-----------------------------------------------------------------------------
// eoChromXover
//-----------------------------------------------------------------------------

class eoChromXover: public eoQuadOp<Chrom>
{
public:
  bool operator()(Chrom& chrom1, Chrom& chrom2)
  {
    chrom1.normalize();
    chrom2.desaturate();

    mse::net tmp1(chrom1), tmp2(chrom2);
    ensure_datasets_initialized();
    tmp1.train(*trn_set, 100, 0, 0.001);
    tmp2.train(*trn_set, 100, 0, 0.001);

    return true;
  }
};

//-----------------------------------------------------------------------------
// eoChromEvaluator
//-----------------------------------------------------------------------------

int correct(const mlp::net& net, const mlp::set& set)
{
  int sum = 0;

  for (mlp::set::const_iterator s = set.begin(); s != set.end(); ++s)
    {
      unsigned partial = 0;

      for (unsigned i = 0; i < s->output.size(); ++i)
	if ((s->output[i] < 0.5 && net(s->input)[i] < 0.5) ||
	    (s->output[i] > 0.5 && net(s->input)[i] > 0.5))
	  ++partial;

      if (partial == s->output.size())
	++sum;
    }

  return sum;
}

phenotype eoChromEvaluator(const Chrom& chrom)
{
  phenotype p;
  ensure_datasets_initialized();
  p.trn_ok = correct(chrom, *trn_set);
  p.val_ok = correct(chrom, *val_set);
  p.tst_ok = correct(chrom, *tst_set);
  p.mse_error = mse::error(chrom, *val_set);

  return p;
}

//-----------------------------------------------------------------------------

#endif // gprop_h

// Local Variables:
// mode:C++
// End:
