template <class EOT>
peoMoeoPop<EOT>&  do_make_pop(eoParser & _parser, eoState& _state, eoInit<EOT> & _init)
{
  // random seed
    eoValueParam<uint32_t>& seedParam = _parser.createParam(uint32_t(0), "seed", "Random number seed", 'S');
    if (seedParam.value() == 0)
	seedParam.value() = time(0);
    eoValueParam<unsigned>& popSize = _parser.createParam(unsigned(20), "popSize", "Population Size", 'P', "Evolution Engine");

  // Either load or initialize
  // create an empty pop and let the state handle the memory
  peoMoeoPop<EOT>& pop = _state.takeOwnership(peoMoeoPop<EOT>());

  eoValueParam<std::string>& loadNameParam = _parser.createParam(std::string(""), "Load","A save file to restart from",'L', "Persistence" );
  eoValueParam<bool> & recomputeFitnessParam = _parser.createParam(false, "recomputeFitness", "Recompute the fitness after re-loading the pop.?", 'r',  "Persistence" );

  if (loadNameParam.value() != "") // something to load
    {
      // create another state for reading
      eoState inState;		// a state for loading - WITHOUT the parser
      // register the rng and the pop in the state, so they can be loaded,
      // and the present run will be the exact continuation of the saved run
      // eventually with different parameters
      inState.registerObject(pop);
      inState.registerObject(rng);
      inState.load(loadNameParam.value()); //  load the pop and the rng
      // the fitness is read in the file:
      // do only evaluate the pop if the fitness has changed
      if (recomputeFitnessParam.value())
	{
	  for (unsigned i=0; i<pop.size(); i++)
	    pop[i].invalidate();
	}
      if (pop.size() < popSize.value())
	std::cerr << "WARNING, only " << pop.size() << " individuals read in file " << loadNameParam.value() << "\nThe remaining " << popSize.value() - pop.size() << " will be randomly drawn" << std::endl;
      if (pop.size() > popSize.value())
	{
	  std::cerr << "WARNING, Load file contained too many individuals. Only the best will be retained" << std::endl;
	  pop.resize(popSize.value());
	}
    }
  else				// nothing loaded from a file
    {
      rng.reseed(seedParam.value());
    }

  if (pop.size() < popSize.value()) // missing some guys
    {
      // Init pop from the randomizer: need to use the append function
      pop.append(popSize.value(), _init);
    }

  // for future stateSave, register the algorithm into the state
  _state.registerObject(_parser);
  _state.registerObject(pop);
  _state.registerObject(rng);

  return pop;
}
