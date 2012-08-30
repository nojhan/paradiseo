template<template <class> class EOAlgo, class EOT, class Policy>
void paradiseo::smp::MWModel<EOAlgo,EOT,Policy>::operator()(eoPop<EOT>& _pop, const eoSyncEasyPSO_tag&) 
{
    try
    {
        // initializes the topology, velocity, best particle(s)
        this->init();

        // just to use a loop eval
        eoPop<EOT> empty_pop;

        do
        {
            // perform velocity evaluation
            //scheduler(this->velocity, _pop);
            this->velocity.apply(_pop);

            // apply the flight
            this->flight.apply(_pop);

            // evaluate the position (with a loop eval, empty_swarm IS USELESS)
            scheduler(EOAlgo<EOT>::eval, _pop);

            // update the topology (particle and local/global best(s))
            this->velocity.updateNeighborhood(_pop);

        } while (this->continuator(_pop));

    }
    catch (std::exception & e)
    {
        std::string s = e.what ();
        s.append (" in eoSyncEasyPSO");
        throw std::runtime_error (s);
    }
}
