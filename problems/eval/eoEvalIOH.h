
#ifndef _eoEvalIOH_h
#define _eoEvalIOH_h

#include <ioh.hpp>

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

        eoEvalIOHproblem(ioh::problem::Problem<ScalarType> & pb) :
                _ioh_pb(&pb),
                _has_log(false),
                _ioh_log(nullptr)
        { }

        eoEvalIOHproblem(ioh::problem::Problem<ScalarType> & pb, ioh::Logger & log ) :
                _ioh_pb(&pb),
                _has_log(true),
                _ioh_log(&log)
       {
           // _ioh_log->track_problem(*_ioh_pb);
           pb.attach_logger(log);
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
        void problem(ioh::problem::Problem<ScalarType> & pb )
        {
            _ioh_pb = &pb;
            // _ioh_log->track_problem(pb);
            _ioh_pb->attach_logger(_ioh_log);
        }

        bool has_logger() const {return _has_log;}

        ioh::Logger & logger() {return *_ioh_log;}

    protected:
        ioh::problem::Problem<ScalarType> * _ioh_pb;

        bool _has_log;
        ioh::Logger * _ioh_log;

        virtual Fitness call(EOT& sol)
        {
            Fitness f = (*_ioh_pb)(sol);
            if(_has_log) {
                _ioh_log->log(_ioh_pb->log_info());
            }
            return f;
        }
};


// /** Wrap an IOHexperimenter's suite class within an eoEvalFunc. Useful for algorithm selection.
//  *
//  * WARNING: only handle a suite of problems of A UNIQUE, SINGLE DIMENSION.
//  * Because a given eoAlgo is bond to a instanciated eoInit (most probably an eoInitWithDim)
//  * which is parametrized with a given dimension.
//  *
//  * The idea is to run the given algorithm on a whole suite of problems
//  * and output its aggregated performance.
//  *
//  * See https://github.com/IOHprofiler/IOHexperimenter
//  *
//  * The main template EOT defines the interface of this functor,
//  * that is how the algorithm instance is encoded
//  * (e.g. an eoAlgoFoundry's integer vector).
//  * The SUBEOT template defines the encoding of the sub-problem,
//  * which the encoded algorithm have to solve
//  * (e.g. a OneMax problem).
//  *
//  * @note: This will not reset the given pop between two calls
//  * of the given algorithm on new problems.
//  * You most probably want to wrap your algorithm
//  * in an eoAlgoRestart to do that for you.
//  *
//  * Handle only IOH experimeneter `stat` classes which template type STAT
//  * is explicitely convertible to the given fitness.
//  * Any scalar is most probably already convertible, but compound classes
//  * (i.e. for multi-objective problems) are most probàbly not.
//  *
//  * @note: You're responsible of adding a conversion operator
//  * to the given STAT type, if necessary
//  * (this is checked by a static assert in the constructor).
//  *
//  * @note: You're also responsible of matching the fitness' encoding scalar type
//  * (IOH handle double and int, as of 2020-03-09).
//  *
//  * You will need to pass the IOH include directory to your compiler
//  * (e.g. IOHexperimenter/build/Cpp/src/).
//  */
// template<class EOT, class SUBEOT, class STAT>
// class eoEvalIOHsuiteSingleDim : public eoEvalFunc<EOT>
// {
//     public:
//         using EOType = EOT;
//         using Fitness = typename EOType::Fitness;
//         using ScalarType = typename Fitness::ScalarType;

//         /** Takes an ecdf_logger that computes the base data structure
//          * on which a ecdf_stat will be called to compute an
//          * aggregated performance measure, which will be the evaluated fitness.
//          *
//          * As such, the logger and the stat are mandatory.
//          *
//          * @note: The given logger should be at least embedded
//          * in the logger bound with the given eval.
//          */
//         eoEvalIOHsuiteSingleDim(
//                 eoEvalIOHproblem<SUBEOT>& eval,
//                 eoAlgoFoundry<SUBEOT>& algo,
//                 eoPop<SUBEOT>& pop,
//                 ioh::suite::Suite<ScalarType>& suite,
//                 ioh::logger::ECDF<ScalarType>& log,
//                 ioh::logger::ECDFStat<STAT>& stat
//             ) :
//                 _eval(eval),
//                 _algo(algo),
//                 _pop(pop),
//                 _ioh_suite(&suite),
//                 _ioh_log(log),
//                 _ioh_stat(stat)
//        {
//            static_assert(std::is_convertible<STAT,Fitness>::value);
//            assert(eval.has_log());
//            _ioh_log.target_suite(suite);
//        }

//         virtual void operator()(EOType& sol)
//         {
//             if(not sol.invalid()) {
//                 return;
//             }

//             sol.fitness( call( sol ) );
//         }

//         /** Update the suite pointer for a new one.
//          *
//          * This is useful if you assembled a ParadisEO algorithm
//          * and call it several time in an IOHexperimenter's loop across several suites.
//          * Instead of re-assembling your algorithm,
//          * just update the suite pointer.
//          */
//         void suite( ioh::suite::Suite<ScalarType> & suite )
//         {
//             _ioh_suite = &suite;
//             _ioh_log.target_suite(suite);
//         }

//     protected:
//         //! Sub-problem  @{
//         eoEvalIOHproblem<SUBEOT>& _eval;
//         eoAlgoFoundry<SUBEOT>& _algo;
//         eoPop<SUBEOT>& _pop;
//         //! @}

//         //! IOH @{
//         ioh::suite::Suite<ScalarType> * _ioh_suite;
//         ioh::logger::Observer<ScalarType> & _ioh_log;
//         ioh::logger::ECDFStat<STAT>& _ioh_stat;
//         //! @}

//         virtual Fitness call(EOType& sol)
//         {
//             // Decode the algorithm encoded in sol.
//             _algo = sol;

//             // Evaluate the performance of the encoded algo instance
//             // on a whole IOH suite benchmark.
//             typename ioh::suite::Suite<ScalarType>::Problem_ptr pb;
//             while( (pb = _ioh_suite->get_next_problem()) ) {

//                 // Consider a new problem.
//                 _eval.problem(*pb); // Will call logger's target_problem.

//                 // Actually solve it.
//                 _algo(_pop); // Will call the logger's write_line.
//                 // There's no need to get back the best fitness from ParadisEO,
//                 // because everything is captured on-the-fly by IOH experimenter.
//             }

//             // Get back the evaluated performance.
//             // The explicit cast from STAT to Fitness which should exists.
//             return static_cast<Fitness>(_ioh_stat(_ioh_log.data()));
//         }
// };


// /** Operator that is called before search for each problem within an IOH suite.
//  *
//  * You most probably need to reinstanciate some operators within your algorithm:
//  * at least the operators depending on the dimension,
//  * as it will change between two calls.
//  *
//  * By providing an operator using this interface,
//  * you can have access to all the information needed to do so.
//  */
// template<class EOT>
// class eoIOHSetup : public eoFunctorBase
// {
//     public:
//         using AtomType = typename EOT::AtomType;
//         virtual void operator()(eoPop<EOT>& pop, typename ioh::suite::Suite<AtomType>::Problem_ptr pb) = 0;
// };

// /** Wrap an IOHexperimenter's suite class within an eoEvalFunc. Useful for algorithm selection.
//  *
//  * The idea is to run the given algorithm on a whole suite of problems
//  * and output its aggregated performance.
//  *
//  * See https://github.com/IOHprofiler/IOHexperimenter
//  *
//  * The main template EOT defines the interface of this functor,
//  * that is how the algorithm instance is encoded
//  * (e.g. an eoAlgoFoundry's integer vector).
//  * The SUBEOT template defines the encoding of the sub-problem,
//  * which the encoded algorithm have to solve
//  * (e.g. a OneMax problem).
//  *
//  * @note: This will not reset the given pop between two calls
//  * of the given algorithm on new problems.
//  * You most probably want to wrap your algorithm
//  * in an eoAlgoRestart to do that for you.
//  *
//  * Handle only IOHprofiler `stat` classes which template type STAT
//  * is explicitely convertible to the given fitness.
//  * Any scalar is most probably already convertible, but compound classes
//  * (i.e. for multi-objective problems) are most probàbly not.
//  *
//  * @note: You're responsible of adding a conversion operator
//  * to the given STAT type, if necessary
//  * (this is checked by a static assert in the constructor).
//  *
//  * @note: You're also responsible of matching the fitness' encoding scalar type
//  * (IOH handle double and int, as of 2020-03-09).
//  *
//  * You will need to pass the IOH include directory to your compiler
//  * (e.g. IOHexperimenter/build/Cpp/src/).
//  */
// template<class EOT, class SUBEOT, class STAT>
// class eoEvalIOHsuite : public eoEvalFunc<EOT>
// {
//     public:
//         using Fitness = typename EOT::Fitness;
//         using ScalarType = typename Fitness::ScalarType;
//         using SubAtomType = typename SUBEOT::AtomType;

//         /** Takes an ecdf_logger that computes the base data structure
//          * on which a ecdf_stat will be called to compute an
//          * aggregated performance measure, which will be the evaluated fitness.
//          *
//          * As such, the logger and the stat are mandatory.
//          *
//          * @note: The given logger should be at least embedded
//          * in the logger thas is bound with the given eval.
//          */
//         eoEvalIOHsuite(
//                 eoEvalIOHproblem<SUBEOT>& eval,
//                 eoAlgoFoundry<SUBEOT>& foundry,
//                 eoPop<SUBEOT>& pop,
//                 eoIOHSetup<SUBEOT>& setup,
//                 ioh::suite::Suite<SubAtomType>& suite,
//                 ioh::logger::ECDF<SubAtomType>& log,
//                 ioh::logger::ECDFStat<STAT>& stat
//             ) :
//                 _eval(eval),
//                 _foundry(foundry),
//                 _pop(pop),
//                 _setup(setup),
//                 _ioh_suite(&suite),
//                 _ioh_log(log),
//                 _ioh_stat(stat)
//        {
//            static_assert(std::is_convertible<STAT,Fitness>::value);
//            assert(_eval.has_logger());
//            _ioh_log.track_suite(suite);
//        }

//         virtual void operator()(EOT& sol)
//         {
//             if(not sol.invalid()) {
//                 return;
//             }

//             sol.fitness( call( sol ) );
//         }

//         /** Update the suite pointer for a new one.
//          *
//          * This is useful if you assembled a ParadisEO algorithm
//          * and call it several time in an IOHexperimenter's loop across several suites.
//          * Instead of re-assembling your algorithm,
//          * just update the suite pointer.
//          */
//         void suite( ioh::suite::Suite<SubAtomType> & suite )
//         {
//             _ioh_suite = &suite;
//             _ioh_log.target_suite(suite);
//         }

//     protected:
//         eoEvalIOHproblem<SUBEOT>& _eval;
//         eoAlgoFoundry<SUBEOT>& _foundry;
//         eoPop<SUBEOT>& _pop;
//         eoIOHSetup<SUBEOT>& _setup;

//         ioh::suite::Suite<SubAtomType> * _ioh_suite;
//         ioh::logger::ECDF<SubAtomType> & _ioh_log;
//         ioh::logger::ECDFStat<STAT>& _ioh_stat;

//         virtual Fitness call(EOT& sol)
//         {
//             // Select an algorithm in the foundry
//             // from the given encoded solution.
//             std::vector<size_t> encoding;
//             std::transform(std::begin(sol), std::end(sol), std::back_inserter(encoding),
//                     [](const SubAtomType& v) -> size_t {return static_cast<size_t>(std::floor(v));} );
//             _foundry.select(encoding);

//             // Evaluate the performance of the encoded algo instance
//             // on a whole IOH suite benchmark.
//             typename ioh::suite::Suite<SubAtomType>::Problem_ptr pb;
//             while( (pb = _ioh_suite->get_next_problem()) ) {

//                 // Setup selected operators.
//                 _setup(_pop, pb);

//                 // Consider a new problem.
//                 _eval.problem(*pb); // Will call logger's target_problem.

//                 // Actually solve it.
//                 _foundry(_pop); // Will call the logger's write_line.
//                 // There's no need to get back the best fitness from ParadisEO,
//                 // because everything is captured on-the-fly by IOH experimenter.
//             }

//             // Get back the evaluated performance.
//             // The explicit cast from STAT to Fitness which should exists.
//             return static_cast<Fitness>(_ioh_stat(_ioh_log.data()));
//         }
// };

#endif // _eoEvalIOH_h

