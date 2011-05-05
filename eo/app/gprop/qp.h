//-----------------------------------------------------------------------------
// qp.h
//-----------------------------------------------------------------------------

#ifndef qp_h
#define qp_h

//-----------------------------------------------------------------------------

#include <iostream>               // istream ostream
#include <algorithm>              // fill
#include <vector>                 // vector
#include <utils/rnd_generators.h> // uniform_generator
#include <mlp.h>                  // neuron layer net

//-----------------------------------------------------------------------------

namespace qp
{
  //---------------------------------------------------------------------------
  // useful typedefs
  //---------------------------------------------------------------------------

  using mlp::real;
  using mlp::vector;

  using mlp::max_real;
  using mlp::min_real;

  using mlp::set;

  //---------------------------------------------------------------------------
  // useful constants
  //---------------------------------------------------------------------------

  const real eta_default    = 0.5;
  const real eta_floor      = 0.0001;
  const real alpha_default  = 0.9;
  const real lambda_default = 0.5;
  const real lambda0        = 0.1;
  const real backtrack_step = 0.5;
  const real me_floor       = 0.0001;
  const real mw_floor       = 0.0001;


  //---------------------------------------------------------------------------
  // neuron
  //---------------------------------------------------------------------------

  struct neuron
  {
    mlp::neuron* n;
    real out, delta, ndelta, dbias1, dbias2;
    vector dweight1, dweight2, dxo;

    neuron(mlp::neuron& _n):
      n(&_n), out(0), delta(0), ndelta(0), dbias1(0), dbias2(0),
      dweight1(n->weight.size(), 0),
      dweight2(n->weight.size(), 0),
      dxo(n->weight.size(), 0) {}

    void reset()
    {
      // underlaying neuron
      n->reset();

      // addons
      out = delta = ndelta = dbias1 = dbias2 = 0;
      fill(dweight1.begin(), dweight1.end(), 0);
      fill(dweight2.begin(), dweight2.end(), 0);
      fill(dxo.begin(), dxo.end(), 0);
    }

    real operator()(const vector& input)
    {
      return out = mlp::sigmoid(n->bias + dbias1 +
				(n->weight + dweight1) * input);
    }
  };

  std::ostream& operator<<(std::ostream& os, const neuron& n)
  {
    return os << *n.n << " " << n.out << " " << n.delta << " "
	      << n.ndelta << " " << n.dbias1 << " " << n.dbias2 << " "
	      << n.dweight1 << " " << n.dweight2 << " " << n.dxo;
  }


  //---------------------------------------------------------------------------
  // layer
  //---------------------------------------------------------------------------

  class layer: public std::vector<neuron>
  {
  public:
    layer(mlp::layer& l)//: std::vector<neuron>(l.begin(), l.end()) {}
    {
      for (mlp::layer::iterator n = l.begin(); n != l.end(); ++n)
	push_back(neuron(*n));
    }

    void reset()
    {
      for(iterator n = begin(); n != end(); ++n)
	n->reset();
    }

    vector operator()(const vector& input)
    {
      vector output(size());

      for(unsigned i = 0; i < output.size(); ++i)
	output[i] = (*this)[i](input);

      return output;
    }
  };


  //---------------------------------------------------------------------------
  // net
  //---------------------------------------------------------------------------

  class net: public std::vector<layer>
  {
  public:
    net(mlp::net& n) //: std::vector<layer>(n.begin(), n.end()) { reset(); }
    {
      for (mlp::net::iterator l = n.begin(); l != n.end(); ++l)
	push_back(*l);
    }

    virtual ~net() {}

    void reset()
    {
      for(iterator l = begin(); l != end(); ++l)
	l->reset();
    }

    real train(const set& ts,
	       unsigned   epochs,
	       real       target_error,
	       real       tolerance,
	       real       eta      = eta_default,
	       real       momentum = alpha_default,
	       real       lambda   = lambda_default)
    {
      real error_ = max_real;

      while (epochs-- && error_ > target_error)
	{
	  real last_error = error_;

	  init_delta();

	  error_ = error(ts);

	  if (error_ < last_error + tolerance)
	    {
	      coeff_adapt(eta, momentum, lambda);
	      weight_update(ts.size(), true, eta, momentum);
	    }
	  else
	    {
	      eta *= backtrack_step;
	      eta = std::max(eta, eta_floor);
	      momentum = eta * lambda;
	      weight_update(ts.size(), false, eta, momentum);
	      error_ = last_error;
	    }
	}

      return error_;
    }

    virtual real error(const set& ts) = 0;

    // protected:
    void forward(vector input)
    {
      for (iterator l = begin(); l != end(); ++l)
	{
	  vector tmp = (*l)(input);
	  input.swap(tmp);
	}
    }

    // private:
    void init_delta()
    {
      for (iterator l = begin(); l != end(); ++l)
	for (layer::iterator n = l->begin(); n != l->end(); ++n)
	  fill(n->dxo.begin(), n->dxo.end(), n->ndelta = 0.0);
    }

    void coeff_adapt(real& eta, real& momentum, real& lambda)
    {
      real me = 0, mw = 0, ew = 0;

      for (iterator l = begin(); l != end(); ++l)
	for (layer::iterator n = l->begin(); n != l->end(); ++n)
	  {
	    me += n->dxo * n->dxo;
	    mw += n->dweight1 * n->dweight1;
	    ew += n->dxo * n->dweight1;
	  }

      me = std::max(static_cast<real>(sqrt(me)), me_floor);
      mw = std::max(static_cast<real>(sqrt(mw)), mw_floor);
      eta *= (1.0 + 0.5 * ew / ( me * mw));
      eta = std::max(eta, eta_floor);
      lambda = lambda0 * me / mw;
      momentum = eta * lambda;
#ifdef DEBUG
      std::cout << me << "  \t" << mw << "  \t" << ew << "  \t"
	   << eta << "  \t" << momentum << "  \t" << lambda << std::endl;
#endif // DEBUG
    }

    void weight_update(unsigned size, bool fire, real eta, real momentum)
    {
      for (iterator l = begin(); l != end(); ++l)
	for (layer::iterator n = l->begin(); n != l->end(); ++n)
	  {
	    n->ndelta /= size;
	    n->dxo /= size;
	    if (fire)
	      {
		n->n->weight += n->dweight1;
		n->dweight2 = n->dweight1;
		n->n->bias += n->dbias1;
		n->dbias2 = n->dbias1;
	      }
	    n->dweight1 = eta * n->dxo + momentum * n->dweight2;
	    n->dbias1 = eta * n->ndelta + momentum * n->dbias2;
	  }
    }
  };

  //---------------------------------------------------------------------------

} // namespace qp

//-----------------------------------------------------------------------------

#endif // qp_h

// Local Variables:
// mode:C++
// End:
