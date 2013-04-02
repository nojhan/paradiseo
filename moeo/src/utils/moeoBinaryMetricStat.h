

template <class MOEOT, class T = double>
class moeoBinaryMetricStat : public eoStat<MOEOT, T>
{
public:
    /** The objective vector type of a solution */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;

    moeoBinaryMetricStat(
            moeoVectorVsVectorBinaryMetric<ObjectiveVector, T> & metric,
            std::string description,
            T default_value = 0
        ) :
            eoStat<MOEOT,T>( default_value, description),
            _metric(metric),
            _first_gen(true)
    {}

    virtual std::string className(void) const
        { return "moeoBinaryMetricStat"; }


    virtual void operator()( const eoPop<MOEOT> & pop )
    {
        if( pop.size() ) {
            if( _first_gen ) {
                _first_gen = false;
            } else {
              // creation of the two Pareto sets
              std::vector < ObjectiveVector > from;
              std::vector < ObjectiveVector > to;
              for (unsigned int i=0; i<pop.size(); i++) {
                from.push_back( pop[i].objectiveVector() );
              }
              for (unsigned int i=0 ; i<_prev_pop.size(); i++) {
                to.push_back( _prev_pop[i].objectiveVector() );
              }

              // compute and save
              this->value() = _metric(from,to);
            } // if first gen

            // copy the pop
            _prev_pop = pop;
        } // if pop size
    }

protected:
    /** binary metric comparing two Pareto sets */
    moeoVectorVsVectorBinaryMetric<ObjectiveVector, double> & _metric;

    /** (n-1) population */
    eoPop<MOEOT> _prev_pop;

    /** is it the first generation ? */
    bool _first_gen;

};
