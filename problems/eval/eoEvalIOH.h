
#ifndef _eoEvalIOH_h
#define _eoEvalIOH_h

#include <IOHprofiler_problem.hpp>
#include <IOHprofiler_csv_logger.h>

/** Wrap an IOHexperimenter's problem class within an eoEvalFunc.
 *
 * See https://github.com/IOHprofiler/IOHexperimenter
 *
 * Handle only fitnesses that inherits from eoScalarFitness.
 *
 * @note: You're responsible of matching the fitness' scalar type (IOH handle double and int, as of 2020-03-09).
 *
 * You will need to pass the IOH include directory to your compiler (e.g. IOHexperimenter/build/Cpp/src/).
 */
template<class EOT>
class eoEvalIOH : public eoEvalFunc<EOT>
{
    public:
        using Fitness = typename EOT::Fitness;
        using ScalarType = typename Fitness::ScalarType;

        eoEvalIOH( IOHprofiler_problem<ScalarType> & pb) :
                _ioh_pb(pb),
                _has_log(false)
        { }

        eoEvalIOH( IOHprofiler_problem<ScalarType> & pb, IOHprofiler_csv_logger & log ) :
                _ioh_pb(pb),
                _has_log(true),
                _ioh_log(log)
       { }

        virtual void operator()(EOT& sol)
        {
            if(not sol.invalid()) {
                return;
            }

            sol.fitness( call( sol ) );
        }

    protected:
        IOHprofiler_problem<ScalarType> & _ioh_pb;
        bool _has_log;
        IOHprofiler_csv_logger & _ioh_log;

        virtual Fitness call(EOT& sol)
        {
            Fitness f = _ioh_pb.evaluate(sol);
            if(_has_log) {
                _ioh_log.write_line(_ioh_pb.loggerInfo());
            }
            return f;
        }

};

#endif // _eoEvalIOH_h
