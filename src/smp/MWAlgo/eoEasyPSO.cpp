template<template <class> class EOAlgo, class EOT, class Policy>
void paradiseo::smp::MWModel<EOAlgo,EOT,Policy>::operator()(eoPop<EOT>& _pop, const eoEasyPSO_tag&) 
{
    try
    {
        // initializes the topology, velocity, best particle(s)
        this->init();
        do
        {
            // loop over all the particles for the current iteration
            for (unsigned idx = 0; idx < _pop.size (); idx++)
            {
                // perform velocity evaluation
                this->velocity (_pop[idx],idx);

                // apply the flight
                this->flight (_pop[idx]);
            }

                // evaluate the position
                scheduler(EOAlgo<EOT>::eval, _pop);
                
            for (unsigned idx = 0; idx < _pop.size (); idx++)
                // update the topology (particle and local/global best(s))
                this->velocity.updateNeighborhood(_pop[idx],idx);

        } while (this->continuator (_pop));
    }
    catch (std::exception & e)
    {
        std::string s = e.what ();
        s.append (" in eoEasyPSO");
        throw std::runtime_error (s);
    }
}
