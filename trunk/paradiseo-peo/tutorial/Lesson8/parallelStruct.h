template < class EOT, class TYPE> class moeoSelector : public selector< TYPE >
  {
  public:


    moeoSelector(moeoSelectOne<EOT> & _select, unsigned _nb_select, const TYPE & _source): selector (_select), nb_select(_nb_select), source(_source)
    {}
    
	// Example
	virtual void operator()(TYPE & _dest)
    {
      size_t target = static_cast<size_t>(nb_select);
      _dest.resize(target);
      for (size_t i = 0; i < _dest.size(); ++i)
        _dest[i] = selector(source);
    }

  protected:
    moeoSelectOne<EOT> & selector ;
    unsigned nb_select;
    const TYPE & source;
  };
  
  
  template < class TYPESOUR, class TYPEDEST> class moeoReplaceArchive : public replacement< TYPESOUR >
  {
  public:

    moeoReplaceArchive(TYPEDEST & _destination): destination(_destination)
    {}

	// Example
    virtual void operator()(TYPESOUR & _source)
    {
    	destination.update (_source);
    }

  protected:
    TYPEDEST & destination;
  };
  
  
  
  template < class TYPESOUR, class TYPEDEST> class moeoReplace : public replacement< TYPESOUR >
  {
  public:

    moeoReplace(TYPEDEST & _destination): destination(_destination)
    {}

	 // Example
    virtual void operator()(TYPESOUR & _source)
    {
	    for(unsigned i=0;i<_source.size();i++)
    	{
     		unsigned ind=0;
      		double worst=destination[0].fitness();
      		for (unsigned j=1;j<destination.size();j++)
        		if ( destination[j].fitness()< worst)
          			{
            			ind=j;
            			worst=destination[j].fitness();
          			}
      		destination[ind]=_source[i];
    	}
    }

  protected:
    TYPEDEST & destination;
  };
  
  
template <class TYPE> class moeoSelectorArchive : public selector< TYPE >
  {
  public:


    moeoSelectorArchive(const TYPE & _source): source(_source)
    {}
    
	// Example
	virtual void operator()(TYPE & _dest)
    {
    	_dest.update (source);
    }

  protected:
  	const TYPE & source;
};
