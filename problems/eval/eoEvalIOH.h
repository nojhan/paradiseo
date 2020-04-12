
#ifndef _eoEvalIOH_h
#define _eoEvalIOH_h

#include <IOHprofiler_problem.hpp>
#include <IOHprofiler_observer.hpp>

/** Wrap an IOHexperimenter's problem class within an eoEvalFunc.
 *
 * See https://github.com/IOHprofiler/IOHexperimenter
 *
 * Handle only fitnesses that inherits from eoScalarFitness.
 *
 * @note: You're responsible of matching the fitness' encoding scalar type (IOH handle double and int, as of 2020-03-09).
 * @note: You're responsible of calling `activate_logger` (if necessary), but it will call `target_problem` for you.
 *
 * You will need to pass the IOH include directory to your compiler (e.g. IOHexperimenter/build/Cpp/src/).
 */
template<class EOT>
class eoEvalIOHproblem : public eoEvalFunc<EOT>
{
    public:
        using Fitness = typename EOT::Fitness;
        using ScalarType = typename Fitness::ScalarType;

        eoEvalIOHproblem(IOHprofiler_problem<ScalarType> & pb) :
                _ioh_pb(&pb),
                _has_log(false),
                _ioh_log(nullptr)
        { }

        eoEvalIOHproblem(IOHprofiler_problem<ScalarType> & pb, IOHprofiler_observer<ScalarType> & log ) :
                _ioh_pb(&pb),
                _has_log(true),
                _ioh_log(&log)
       {
           _ioh_log->target_problem(*_ioh_pb);
       }

        virtual void operator()(EOT& sol)
        {
            if(not sol.invalid()) {
                return;
            }

            sol.fitness( call( sol ) );
        }

        /** Update the problem pointer for a new one.
         *
         * This is useful if you assembled a ParadisEO algorithm
         * and call it several time in an IOHexperimenter's suite loop.
         * Instead of re-assembling your algorithm,
         * just update the problem pointer.
         */
        void problem(IOHprofiler_problem<ScalarType> & pb )
        {
            _ioh_pb = &pb;
            _ioh_log->target_problem(pb);
        }

        bool has_logger() const {return _has_log;}

        IOHprofiler_observer<ScalarType> & observer() {return *_ioh_log;}

    protected:
        IOHprofiler_problem<ScalarType> * _ioh_pb;

        bool _has_log;
        IOHprofiler_observer<ScalarType> * _ioh_log;

        virtual Fitness call(EOT& sol)
        {
            Fitness f = _ioh_pb->evaluate(sol);
            if(_has_log) {
                _ioh_log->write_line(_ioh_pb->loggerInfo());
            }
            return f;
        }
};

#endif // _eoEvalIOH_h

