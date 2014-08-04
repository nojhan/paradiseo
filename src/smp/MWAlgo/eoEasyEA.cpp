template<template <class> class EOAlgo, class EOT, class Policy>
void paradiseo::smp::MWModel<EOAlgo,EOT,Policy>::operator()(eoPop<EOT>& _pop, const eoEasyEA_tag&) 
{
    
	if (this->isFirstCall)
	{
		size_t total_capacity = _pop.capacity() + this->offspring.capacity();
		_pop.reserve(total_capacity);
		this->offspring.reserve(total_capacity);
		this->isFirstCall = false;
	}

      eoPop<EOT> empty_pop;

      this->popEval(empty_pop, _pop); // A first eval of pop.

      do
        {
          try
            {
              unsigned pSize = _pop.size();
              this->offspring.clear(); // new offspring
              this->breed(_pop, this->offspring);
              
              scheduler(EOAlgo<EOT>::eval, this->offspring); // eval of parents + offspring if necessary
              
              this->replace(_pop, this->offspring); // after replace, the new pop. is in _pop

              if (pSize > _pop.size())
                throw std::runtime_error("Population shrinking!");
              else if (pSize < _pop.size())
                throw std::runtime_error("Population growing!");

            }
          catch (std::exception& e)
            {
              std::string s = e.what();
              s.append( " in eoEasyEA");
              throw std::runtime_error( s );
            }
        }
      while (this->continuator( _pop ) );
}
