
#ifndef _eoStandardBitMutation_h_
#define _eoStandardBitMutation_h_

#include "../utils/eoRNG.h"

/** Standard bit mutation with mutation rate p:
 * choose k from the binomial distribution Bin(n,p) and apply flip_k(x).
 *
 * @ingroup Bitstrings
 * @ingroup Variators
 */
template<class EOT>
class eoStandardBitMutation : public eoMonOp<EOT>
{
    public:
        eoStandardBitMutation(double rate = 0.5) :
            _rate(rate),
            _nb(1),
            _bitflip(_nb)
        {}

        virtual bool operator()(EOT& chrom)
        {
            _nb = eo::rng.binomial(chrom.size(),_rate);
            // BitFlip operator is bound to the _nb reference,
            // thus one don't need to re-instantiate.
            return _bitflip(chrom);
        }

        virtual std::string className() const {return "eoStandardBitMutation";}

    protected:
        double _rate;
        unsigned _nb;
        eoDetSingleBitFlip<EOT> _bitflip;
};

/** Uniform bit mutation with mutation rate p:
 * choose k from the uniform distribution U(0,n) and apply flip_k(x).
 *
 * @ingroup Bitstrings
 * @ingroup Variators
 */
template<class EOT>
class eoUniformBitMutation : public eoMonOp<EOT>
{
    public:
        eoUniformBitMutation(double rate = 0.5) :
            _rate(rate),
            _nb(1),
            _bitflip(_nb)
        {}

        virtual bool operator()(EOT& chrom)
        {
            _nb = eo::rng.random(chrom.size());
            // BitFlip operator is bound to the _nb reference,
            // thus one don't need to re-instantiate.
            return _bitflip(chrom);
        }

        virtual std::string className() const {return "eoUniformBitMutation";}

    protected:
        double _rate;
        unsigned _nb;
        eoDetSingleBitFlip<EOT> _bitflip;
};


/** Conditional standard bit mutation with mutation rate p:
 * choose k from the binomial distribution Bin(n,p) until k >0
 * and apply flip_k(x).
 *
 * This is identical to sampling k from the conditional binomial
 * distribution Bin>0(n,p) which re-assigns the probability to sample
 * a 0 proportionally to all values i ∈ [1..n]. 
 *
 * @ingroup Bitstrings
 * @ingroup Variators
 */
template<class EOT>
class eoConditionalBitMutation : public eoStandardBitMutation<EOT>
{
    public:
        eoConditionalBitMutation(double rate = 0.5) :
            eoStandardBitMutation<EOT>(rate)
        {}

        virtual bool operator()(EOT& chrom)
        {
            assert(chrom.size()>0);
            this->_nb = eo::rng.binomial(chrom.size()-1,this->_rate);
            this->_nb++;
            // BitFlip operator is bound to the _nb reference,
            // thus one don't need to re-instantiate.
            return this->_bitflip(chrom);
        }

        virtual std::string className() const {return "eoConditionalBitMutation";}
};

/** Shifted standard bit mutation with mutation rate p:
 * choose k from the binomial distribution Bin(n,p).
 * When k= 0, set k= 1. Apply flip_k(x).
 *
 * This is identical to sampling k from the conditional binomial
 * distribution Bin0→1(n,p) which re-assigns the probability to
 * sample a 0 to sampling k= 1.
 *
 * @ingroup Bitstrings
 * @ingroup Variators
 */
template<class EOT>
class eoShiftedBitMutation : public eoStandardBitMutation<EOT>
{
    public:
        eoShiftedBitMutation(double rate = 0.5) :
            eoStandardBitMutation<EOT>(rate)
        {}

        virtual bool operator()(EOT& chrom)
        {
            assert(chrom.size()>0);
            this->_nb = eo::rng.binomial(chrom.size()-1,this->_rate);
            if(this->_nb == 0) {
                this->_nb = 1;
            }
            // BitFlip operator is bound to the _nb reference,
            // thus one don't need to re-instantiate.
            return this->_bitflip(chrom);
        }

        virtual std::string className() const {return "eoShiftedBitMutation";}
};

/** Mutation which size is sample in a gaussian.
 *
 * sample k from the normal distribution N(pn,σ^2)
 * and apply flip_k(x).
 *
 * From:
 * Furong Ye, Carola Doerr, and Thomas Back.
 * Interpolating local and global search by controllingthe variance of standard bit mutation.
 * In 2019 IEEE Congress on Evolutionary Computation(CEC), pages 2292–2299.
 *
 * In contrast to standard bit mutation, this operators allows to scale
 * the variance of the mutation strength independently of the mean.
 *
 * @ingroup Bitstrings
 * @ingroup Variators
 */
template<class EOT>
class eoNormalBitMutation : public eoStandardBitMutation<EOT>
{
    public:
        eoNormalBitMutation(double rate = 0.5, double variance = 1) :
            eoStandardBitMutation<EOT>(rate),
            _variance(variance)
        {}

        virtual bool operator()(EOT& chrom)
        {
            this->_nb = eo::rng.normal(this->_rate * chrom.size(), _variance);
            if(this->_nb >= chrom.size()) {
                this->_nb = eo::rng.random(chrom.size());
            }
            // BitFlip operator is bound to the _nb reference,
            // thus one don't need to re-instantiate.
            return this->_bitflip(chrom);
        }

        virtual std::string className() const {return "eoNormalBitMutation";}

    protected:
        double _variance;
};

/** Fast mutation which size is sampled from an adaptive power law.
 *
 * From:
 * Benjamin Doerr, Huu Phuoc Le, Régis Makhmara, and Ta Duy Nguyen.
 * Fast genetic algorithms.
 * In Proc. of Genetic and Evolutionary Computation Conference (GECCO’17), pages 777–784.ACM, 2017.
 *
 * @ingroup Bitstrings
 * @ingroup Variators
 */
template<class EOT>
class eoFastBitMutation : public eoStandardBitMutation<EOT>
{
    public:
        eoFastBitMutation(double rate = 0.5, double beta = 1.5) :
            eoStandardBitMutation<EOT>(rate),
            _beta(beta)
        {
            assert(beta > 1);
        }

        virtual bool operator()(EOT& chrom)
        {
            this->_nb = powerlaw(chrom.size(),_beta);
            // BitFlip operator is bound to the _nb reference,
            // thus one don't need to re-instantiate.
            return this->_bitflip(chrom);
        }

        virtual std::string className() const {return "eoFastBitMutation";}

    protected:

        double powerlaw(unsigned n, double beta)
        {
            double cnb = 0;
            for(unsigned i=1; i<n; ++i) {
                cnb += std::pow(i,-beta);
            }
            return eo::rng.powerlaw(0,n,beta) / cnb;
        }

        double _beta;
};

#endif // _eoStandardBitMutation_h_
