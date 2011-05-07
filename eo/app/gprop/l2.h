//-----------------------------------------------------------------------------
// l2.h
//-----------------------------------------------------------------------------

#ifndef l2_h
#define l2_h

//-----------------------------------------------------------------------------

#include <math.h>  // log
#include <qp.h>    // neuron layer net set

//-----------------------------------------------------------------------------

namespace l2
{
  //---------------------------------------------------------------------------
  // useful typedefs
  //---------------------------------------------------------------------------

  using qp::real;
  using qp::vector;
  using qp::max_real;
  using qp::min_real;
  using qp::set;
  using qp::neuron;
  using qp::layer;

  //---------------------------------------------------------------------------
  // error
  //---------------------------------------------------------------------------

  real error(const mlp::net& net, const set& ts)
  {
    real error_ = 0.0;

    for (set::const_iterator s = ts.begin(); s != ts.end(); ++s)
      {
	vector out = net(s->input);

	for (unsigned i = 0; i < out.size(); ++i)
	  {
	    real target = s->output[i];
	    real value = out[i];
	    error_ -= target * log(value + min_real) +
	      (1.0 - target) * log(1.0 - value + min_real);
	  }
      }

    return error_;
  }

  //-------------------------------------------------------------------------
  // l2
  //-------------------------------------------------------------------------

  class net: public qp::net
  {
  public:
    net(mlp::net& n): qp::net(n) {}

    real error(const set& ts)
    {
      real error_ = 0;

      for (set::const_iterator s = ts.begin(); s != ts.end(); ++s)
	{
	  forward(s->input);
	  error_ -= backward(s->input, s->output);
	}

      return error_;
    }

  private:
    real backward(const vector& input, const vector& output)
    {
      reverse_iterator current_layer = rbegin();
      reverse_iterator backward_layer = current_layer + 1;
      real error_ = 0;

      // output layer
      for (unsigned j = 0; j < current_layer->size(); ++j)
	{
	  neuron& n = (*current_layer)[j];
	  real out = output[j];
	  n.ndelta += n.delta = (out - n.out) /
	    (n.out * (1.0 - n.out) + min_real) * n.out * (1.0 - n.out);

	  if (size() == 1)                                        // monolayer
	    n.dxo += n.delta * input;
	  else                                                    // multilayer
	    for (unsigned k = 0; k < n.dxo.size(); ++k)
	      n.dxo[k] += n.delta * (*backward_layer)[k].out;

	  error_ += out * log(n.out + min_real) +
	    (1.0 - out) * log(1.0 - n.out + min_real);
	}

      // hidden layers
      while (++current_layer != rend())
	{
	  reverse_iterator forward_layer  = current_layer - 1;
	  reverse_iterator backward_layer = current_layer + 1;

	  for (unsigned j = 0; j < current_layer->size(); ++j)
	    {
	      neuron& n = (*current_layer)[j];
	      real sum = 0;
	      for (unsigned k = 0; k < forward_layer->size(); ++k)
		{
		  neuron& nf = (*forward_layer)[k];
		  sum += nf.delta * (nf.n->weight[j] + nf.dweight1[j]);
		}
	      n.delta = n.out * (1.0 - n.out) * sum;
	      n.ndelta += n.delta;

	      if (backward_layer == rend())               // first hidden layer
		n.dxo += n.delta * input;
	      else                                     // rest of hidden layers
		for (unsigned k = 0; k < n.dxo.size(); ++k)
		  n.dxo[k] += n.delta * (*backward_layer)[k].out;
	    }
	}

      return error_;
    }
  };

  //---------------------------------------------------------------------------

} // namespace l2

//-----------------------------------------------------------------------------

#endif // l2_h

// Local Variables:
// mode:C++
// End:
