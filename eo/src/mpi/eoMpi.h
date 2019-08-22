/*
(c) Thales group, 2012

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
Contact: http://eodev.sourceforge.net

Authors:
    Benjamin Bouvier <benjamin.bouvier@gmail.com>
*/
# ifndef __EO_MPI_H__
# define __EO_MPI_H__

# include <vector>  // std::vector

# include "../utils/eoLogger.h"
# include "../utils/eoTimer.h"
# include "../eoFunctor.h"
# include "../eoExceptions.h"

# include "eoMpiNode.h"
# include "eoMpiAssignmentAlgorithm.h"

namespace eo
{
    /**
     * @ingroup Parallel
     * @defgroup MPI Message Passing Interface
     * @brief See namespace eo::mpi to have all explanations about this module.
     * @{
     */

    /**
     * @brief MPI parallelization helpers for EO.
     *
     * This namespace contains parallelization functions which help to parallelize computations in EO. It is based on a
     * generic algorithm, which is then customized with functors, corresponding to the algorithm main steps. These
     * computations are centralized, i.e there is one central host whose role is to handle the steps of the algorithm ;
     * we call it the "master". The other hosts just have to perform a "dummy" computation, which may be any kind of
     * processing ; we call them, the "slaves", or less pejoratively, the "workers". Workers can communicate to each
     * other, but they receive their orders from the Master and send him back some results. A worker can also be the
     * master of a different parallelization process, as soon as it is a part of its work. Machines of the network, also
     * called hosts, are identified by an unique number: their rank. At any time during the execution of the program,
     * all the hosts know the total number of hosts.
     *
     * A parallelized Job is a set of tasks which are independant (i.e can be executed in random order without
     * modifiying the result) and take a data input and compute a data output to be sent to the Master. The data can be
     * of any type, however they have to be serialized to be sent over a network. It is sufficient that they can be
     * serialized through boost.
     *
     * @todo For serialization purposes, don't depend upon boost. It would be easy to use only eoserial and send strings
     * via mpi.
     *
     * The main steps of the algorithm are the following:
     * - For the master:
     *      - Have we done with the treatment we are doing ?
     *      - If this is the case, we can quit.
     *      - Otherwise, send an input data to some available worker.
     *      - If there's no available worker, wait for a worker to be free.
     *      - When receiving the response, handle it (eventually compute something on the output data, store it...).
     *      - Go back to the first step.
     * - For the worker, it is even easier:
     *      - Wait for an order.
     *      - If there's nothing to do, just quit.
     *      - Otherwise, eventually retrieve data and do the work.
     *      - Go back to the first step.
     *
     * There is of course some network adjustements to do and precisions to give there, but the main ideas are present. As the
     * job is fully centralized, this is the master who tells the workers when to quit and when to work.
     *
     * The idea behind these MPI helpers is to be the most generic possible. If we look back at the steps of the
     * algorithm, we found that the steps can be splitted into 2 parts: the first consists in the steps of any
     * parallelization algorithm and the other consists in the specific parts of the algorithm. Ideally, the user should
     * just have to implement the specific parts of the algorithm. We identified these parts to be:
     * - For the master:
     *      - What does mean to have terminated ? There are only two alternatives, in our binary world: either it is
     *      terminated, or it is not. Hence we only need a function returning a boolean to know if we're done with the
     *      computation : we'll call it IsFinished.
     *      - What do we have to do when we send a task ? We don't have any a priori on the form of the sent data, or
     *      the number of sent data. Moreover, as the tasks are all independant, we don't care of who will do the
     *      computation, as soon as it's done. Knowing the rank of the worker will be sufficient to send him data. We
     *      have identified another function, taking a single argument which is the rank of the worker: we'll call it
     *      SendTask.
     *      - What do we have to do when we receive a response from a worker ? One more time, we don't know which form
     *      or structure can have the receive data, only the user can know. Also we let the user the charge to retrieve
     *      the data ; he just has to know from who the master will retrieve the data. Here is another function, taking
     *      a rank (the sender's one) as a function argument : this will be HandleResponse.
     * - For the worker:
     *      - What is the processing ? It can have any nature. We just need to be sure that a data is sent back to the
     *      master, but it seems difficult to check that: it will be the role of the user to assert that data is sent by
     *      the worker at the end of an execution. We've got identified our last function: ProcessTask.
     *
     * In term of implementation, it would be annoying to have only abstract classes with these 4 methods to implement. It
     * would mean that if you want to alter just one of these 4 functions, you have to implement a new sub class, with a
     * new constructor which could have the same signature. Besides, this fashion doesn't allow you to add dynamic
     * functionalities, using the design pattern Decorator for instance, without implement a class for each type of
     * decoration you want to add. For these reasons, we decided to transform function into functors ; the user can then
     * wrap the existing, basic comportments into more sophisticated computations, whenever he wants, and without the
     * notion of order. We retrieve here the power of extension given by the design pattern Decorator.
     *
     * Our 4 functors could have a big amount of data in common (see eoParallelApply to have an idea).
     * So as to make it easy for the user to implement these 4 functors, we consider that these functors
     * have to share a common data structure. This data structure is referenced (as a pointer) in the 4 functors, so the
     * user doesn't need to pass a lot of parameters to each functor constructor.
     *
     * There are two kinds of jobs:
     * - The job which are launched a fixed and well known amount of times, i.e both master and workers know how many
     *   times they will be launched. They are "one shot jobs".
     * - The job which are launched an unknown amount of times, for instance embedded in a while loop for which we don't
     *   know the amount of repetitions (typically, eoEasyEA loop is a good example, as we don't know the continuator
     *   condition). They are called "multi job".
     * As the master tells the workers to quit, we have to differentiate these two kinds of jobs. When the job is of the
     * kind "multi job", the workers would have to perform a while(true) loop so as to receive the orders ; but even if
     * the master tells them to quit, they would begin another job and wait for another order, while the master would
     * have quit: this would cause a deadlock and workers processes would be blocked, waiting for an order.
     */
    namespace mpi
    {
        /**
         * @brief A timer which allows user to generate statistics about computation times.
         */
        extern eoTimerStat timerStat;

        /**
         * @brief Tags used in MPI messages for framework communication
         *
         * These tags are used for framework communication and fits "channels", so as to differentiate when we're
         * sending an order to a worker (Commands) or data (Messages). They are not reserved by the framework and can be
         * used by the user, but he is not bound to.
         *
         * @ingroup MPI
         */
        namespace Channel
        {
            extern const int Commands;
            extern const int Messages;
        }

        /**
         * @brief Simple orders used by the framework.
         *
         * These orders are sent by the master to the workers, to indicate to them if they should receive another task
         * to do (Continue), if an one shot job is done (Finish) or if a multi job is done (Kill).
         *
         * @ingroup MPI
         */
        namespace Message
        {
            extern const int Continue;
            extern const int Finish;
            extern const int Kill;
        }

        /**
         * @brief If the job only has one master, the user can use this constant, so as not to worry with integer ids.
         *
         * @ingroup MPI
         */
        extern const int DEFAULT_MASTER;

        /**
         * @brief Base class for the 4 algorithm functors.
         *
         * This class can embed a data (JobData) and a wrapper, so as to make all the 4 functors wrappable.
         * We can add a wrapper at initialization or at any time when executing the program.
         *
         * According to RAII, the boolean needDelete helps to know if we have to use the operator delete on the wrapper
         * or not. Hence, if any functor is wrapped, user has just to put this boolean to true, to indicate to wrapper
         * that it should call delete. This allows to mix wrapper initialized in the heap (with new) or in the stack.
         *
         * @param JobData a Data type, which can have any form. It can a struct, a single int, anything.
         *
         * @param Wrapped the type of the functor, which will be stored as a pointer under the name _wrapped.
         * This allows to wrap directly the functor in functors of the same type
         * here, instead of dealing with SharedDataFunction* that we would have to cast all the time.
         * Doing also allows to handle the wrapped functor as the functor we're writing, when coding the wrappers,
         * instead of doing some static_cast. For instance, if there are 2 functors subclasses, fA and fB, fA
         * implementing doFa() and fB implementing doFb(), we could have the following code:
         * @code
         * struct fA_wrapper
         * {
         *   // some code
         *   void doFa()
         *   {
         *      _wrapped->doFa();
         *      std::cout << "I'm a fA wrapper!" << std::endl;
         *      // if we didn't have the second template parameter, but a SharedDataFunction, we would have to do this:
         *      static_cast<fA*>(_wrapped)->doFa();
         *      // do other things (it's a wrapper)
         *   }
         * };
         *
         * struct fB_wrapper
         * {
         *  // some code
         *  void doFb()
         *  {
         *      _wrapped->doFb(); // and not: static_cast<fB*>(_wrapped)->doFb();
         *  }
         * };
         * @endcode
         * This makes the code easier to write for the user.
         *
         * @ingroup MPI
         */
        template< typename JobData, typename Wrapped >
        struct SharedDataFunction
        {
            /**
             * @brief Default constructor.
             *
             * The user is not bound to give a wrapped functor.
             */
            SharedDataFunction( Wrapped * w = 0 ) : _data( 0 ), _wrapped( w ), _needDelete( false )
            {
                // empty
            }

            /**
             * @brief Destructor.
             *
             * Calls delete on the wrapped function, only if necessary.
             */
            virtual ~SharedDataFunction()
            {
                if( _wrapped && _wrapped->needDelete() )
                {
                    delete _wrapped;
                }
            }

            /**
             * @brief Setter for the wrapped function.
             *
             * It doesn't do anything on the current wrapped function, like deleting it.
             */
            void wrapped( Wrapped * w )
            {
                _wrapped = w;
            }

            /**
             * @brief Setter for the data present in the functor.
             *
             * Calls the setter on the functor and on the wrapped functors, in a Composite pattern fashion.
             */
            void data( JobData* d )
            {
                _data = d;
                if( _wrapped )
                {
                    _wrapped->data( d );
                }
            }

            /**
             * @brief Returns true if we need to use operator delete on this wrapper, false otherwise.
             *
             * Allows the user to reject delete responsability to the framework, by setting this value to true.
             **/
            bool needDelete() { return _needDelete; }
            void needDelete( bool b ) { _needDelete = b; }

            protected:
            JobData* _data;
            Wrapped* _wrapped; // Pointer and not a reference so as to be set at any time and to avoid affectation
            bool _needDelete;
        };

        /**
         * @brief Functor (master side) used to send a task to the worker.
         *
         * The user doesn't have to know which worker will receive a task, so we just indicate to master the rank of the
         * worker. The data used for computation have to be explicitly sent by the master to the worker, with indicated
         * rank. Once this functor has been called, the worker is considered busy until it sends a return message to the
         * master.
         *
         * This is a functor implementing void operator()(int), and also a shared data function, containing wrapper on its
         * own type.
         *
         * @ingroup MPI
         */
        template< typename JobData >
        struct SendTaskFunction : public eoUF<int, void>, public SharedDataFunction< JobData, SendTaskFunction<JobData> >
        {
            public:

            SendTaskFunction( SendTaskFunction<JobData>* w = 0 ) : SharedDataFunction<JobData, SendTaskFunction<JobData> >( w )
            {
                // empty
            }

            virtual ~SendTaskFunction() {} // for inherited classes
        };

        /**
         * @brief Functor (master side) used to indicate what to do when receiving a response.
         *
         * The master calls this function as soon as it receives some data, in some channel. Thanks to MPI, we retrieve
         * the rank of the data's sender. This functor is then called with this rank. There is no memoization of a link
         * between sent data and rank, so the user has to implement it, if he needs it.
         *
         * This is a functor implementing void operator()(int), and also a shared data function, containing wrapper on
         * its own type.
         *
         * The master has to receive worker's data on channel (= MPI tag) eo::mpi::Channel::Messages. No other tags are
         * allowed.
         *
         * @ingroup MPI
         */
        template< typename JobData >
        struct HandleResponseFunction : public eoUF<int, void>, public SharedDataFunction< JobData, HandleResponseFunction<JobData> >
        {
            public:

            HandleResponseFunction( HandleResponseFunction<JobData>* w = 0 ) : SharedDataFunction<JobData, HandleResponseFunction<JobData> >( w )
            {
                // empty
            }

            virtual ~HandleResponseFunction() {} // for inherited classes
        };

        /**
         * @brief Functor (worker side) implementing the processing to do.
         *
         * This is where the real computation happen.
         * Whenever the master sends the command "Continue" to workers, which indicates the worker will receive a task,
         * the worker calls this functor. The user has to explicitly retrieve the data, handle it and transmit it,
         * processed, back to the master. Data sent back needs to be transmitted via channel (= MPI tag)
         * eo::mpi::Channel::Messages, and no one else. If the worker does not send any data back to the master, the latter will
         * consider the worker isn't done and a deadlock could occur.
         *
         * This is a functor implementing void operator()(), and also a shared data function, containing wrapper on its
         * own type.
         *
         * @ingroup MPI
         */
        template< typename JobData >
        struct ProcessTaskFunction : public eoF<void>, public SharedDataFunction< JobData, ProcessTaskFunction<JobData> >
        {
            public:

            ProcessTaskFunction( ProcessTaskFunction<JobData>* w = 0 ) : SharedDataFunction<JobData, ProcessTaskFunction<JobData> >( w )
            {
                // empty
            }

            virtual ~ProcessTaskFunction() {} // for inherited classes
        };

        /**
         * @brief Functor (master side) indicating whether the job is done or not.
         *
         * The master loops on this functor to know when to stop. When this functor returns true, the master will wait
         * for the last responses and properly stops the job. Whenever this functor returns false, the master will send
         * tasks, until this functor returns true.
         *
         * This is a functor implementing bool operator()(), and also a shared function, containing wrapper on its own
         * type.
         *
         * @ingroup MPI
         */
        template< typename JobData >
        struct IsFinishedFunction : public eoF<bool>, public SharedDataFunction< JobData, IsFinishedFunction<JobData> >
        {
            public:

            IsFinishedFunction( IsFinishedFunction<JobData>* w = 0 ) : SharedDataFunction<JobData, IsFinishedFunction<JobData> >( w )
            {
                // empty
            }

            virtual ~IsFinishedFunction() {} // for inherited classes
        };

        /**
         * @brief Contains all the required data and the functors to launch a job.
         *
         * Splitting the functors and data from the job in itself allows to use the same functors and data for multiples
         * instances of the same job. You define your store once and can use it a lot of times during your program. If
         * the store was included in the job, you'd have to give again all the functors and all the datas to each
         * invokation of the job.
         *
         * Job store contains the 4 functors (pointers, so as to be able to wrap them ; references couldn't have
         * permitted that) described above and the JobData used by all these functors. It contains
         * also helpers to easily wrap the functors, getters and setters on all of them.
         *
         * The user has to implement data(), which is the getter for retrieving JobData. We don't have any idea of who
         * owns the data, moreover it is impossible to initialize it in this generic JobStore, as we don't know its
         * form. As a matter of fact, the user has to define this in the JobStore subclasses.
         *
         * @ingroup MPI
         */
        template< typename JobData >
        struct JobStore
        {
            /**
             * @brief Default ctor with the 4 functors.
             */
            JobStore(
                SendTaskFunction<JobData>* stf,
                HandleResponseFunction<JobData>* hrf,
                ProcessTaskFunction<JobData>* ptf,
                IsFinishedFunction<JobData>* iff
            ) :
                _stf( stf ), _hrf( hrf ), _ptf( ptf ), _iff( iff )
            {
                // empty
            }

            /**
             * @brief Empty ctor, useful for not forcing users to call the other constructor.
             *
             * When using this constructor, the user have to care about the 4 functors pointers, otherwise null pointer
             * segfaults have to be expected.
             */
            JobStore()
            {
                // empty
            }

            /**
             * @brief Default destructor.
             *
             * JobStore is the highest layer which calls needDelete on its functors.
             */
            ~JobStore()
            {
                if( _stf->needDelete() ) delete _stf;
                if( _hrf->needDelete() ) delete _hrf;
                if( _ptf->needDelete() ) delete _ptf;
                if( _iff->needDelete() ) delete _iff;
            }

            // Getters
            SendTaskFunction<JobData> & sendTask() { return *_stf; }
            HandleResponseFunction<JobData> & handleResponse() { return *_hrf; }
            ProcessTaskFunction<JobData> & processTask() { return *_ptf; }
            IsFinishedFunction<JobData> & isFinished() { return *_iff; }

            // Setters
            void sendTask( SendTaskFunction<JobData>* stf )
            {
                if( !stf )
                    return;

                if( _stf && _stf->needDelete() )
                {
                    delete _stf;
                }
                _stf = stf;
            }

            void handleResponse( HandleResponseFunction<JobData>* hrf )
            {
                if( !hrf )
                    return;

                if( _hrf && _hrf->needDelete() )
                {
                    delete _hrf;
                }
                _hrf = hrf;
            }

            void processTask( ProcessTaskFunction<JobData>* ptf )
            {
                if( !ptf )
                    return;

                if( _ptf && _ptf->needDelete() )
                {
                    delete _ptf;
                }
                _ptf = ptf;
            }

            void isFinished( IsFinishedFunction<JobData>* iff )
            {
                if( !iff )
                    return;

                if( _iff && _iff->needDelete() )
                {
                    delete _iff;
                }
                _iff = iff;
            }

            /**
             * @brief Helpers for wrapping send task functor.
             */
            void wrapSendTask( SendTaskFunction<JobData>* stf )
            {
                if( stf )
                {
                    stf->wrapped( _stf );
                    _stf = stf;
                }
            }

            /**
             * @brief Helpers for wrapping handle response functor.
             */
            void wrapHandleResponse( HandleResponseFunction<JobData>* hrf )
            {
                if( hrf )
                {
                    hrf->wrapped( _hrf );
                    _hrf = hrf;
                }
            }

            /**
             * @brief Helpers for wrapping process task functor.
             */
            void wrapProcessTask( ProcessTaskFunction<JobData>* ptf )
            {
                if( ptf )
                {
                    ptf->wrapped( _ptf );
                    _ptf = ptf;
                }
            }

            /**
             * @brief Helpers for wrapping is finished functor.
             */
            void wrapIsFinished( IsFinishedFunction<JobData>* iff )
            {
                if( iff )
                {
                    iff->wrapped( _iff );
                    _iff = iff;
                }
            }

            virtual JobData* data() = 0;

            protected:

            SendTaskFunction< JobData >* _stf;
            HandleResponseFunction< JobData >* _hrf;
            ProcessTaskFunction< JobData >* _ptf;
            IsFinishedFunction< JobData >* _iff;
        };

        /**
         * @example t-mpi-wrapper.cpp
         */

        /**
         * @brief Class implementing the centralized job algorithm.
         *
         * This class handles all the job algorithm. With its store and its assignment (scheduling) algorithm, it
         * executes the general algorithm described above, adding some networking, so as to make the global process
         * work. It initializes all the functors with the data, then launches the main loop, indicating to workers when
         * they will have to work and when they will finish, by sending them a termination message (integer that can be
         * customized). As the algorithm is centralized, it is also mandatory to indicate what is the MPI rank of the
         * master process, hence the workers will know from who they should receive their commands.
         *
         * Any of the 3 master functors can launch exception, it will be catched and rethrown as a std::runtime_exception
         * to the higher layers.
         *
         * @ingroup MPI
         */
        template< class JobData >
        class Job
        {
            public:
                /**
                 * @brief Main constructor for Job.
                 *
                 * @param _algo The used assignment (scheduling) algorithm. It has to be initialized, with its maximum
                 * possible number of workers (some workers referenced in this algorithm shouldn't be busy). See
                 * AssignmentAlgorithm for more details.
                 *
                 * @param _masterRank The MPI rank of the master.
                 *
                 * @param _workerStopCondition Number of the message which will cause the workers to terminate. It could
                 * be one of the constants defined in eo::mpi::Commands, or any other integer. The user has to be sure
                 * that a message containing this integer will be sent to each worker on the Commands channel, otherwise
                 * deadlock will happen. Master sends Finish messages at the end of a simple job, but as a job can
                 * happen multiples times (multi job), workers don't have to really finish on these messages but on
                 * another message. This is here where you can configurate it. See also OneShotJob and MultiJob.
                 *
                 * @param store The JobStore containing functors and data for this job.
                 */
                Job( AssignmentAlgorithm& _algo,
                     int _masterRank,
                     int _workerStopCondition,
                     JobStore<JobData> & _store
                    ) :
                    assignmentAlgo( _algo ),
                    masterRank( _masterRank ),
                    workerStopCondition( _workerStopCondition ),
                    comm( Node::comm() ),
                    // Functors
                    store( _store ),
                    sendTask( _store.sendTask() ),
                    handleResponse( _store.handleResponse() ),
                    processTask( _store.processTask() ),
                    isFinished( _store.isFinished() )
                {
                    _isMaster = Node::comm().rank() == _masterRank;

                    sendTask.data( _store.data() );
                    handleResponse.data( _store.data() );
                    processTask.data( _store.data() );
                    isFinished.data( _store.data() );
                }

            protected:

                /**
                 * @brief Finally block of the main algorithm
                 *
                 * Herb Sutter's trick for having a finally block, in a try/catch section: invoke a class at the
                 * beginning of the try, its destructor will be called in every cases.
                 *
                 * This implements the end of the master algorithm:
                 * - sends to all available workers that they are free,
                 * - waits for last responses, handles them and sends termination messages to last workers.
                 */
                struct FinallyBlock
                {
                    FinallyBlock(
                            int _totalWorkers,
                            AssignmentAlgorithm& _algo,
                            Job< JobData > & _that
                            ) :
                        totalWorkers( _totalWorkers ),
                        assignmentAlgo( _algo ),
                        that( _that ),
                        // global field
                        comm( Node::comm() )
                    {
                        // empty
                    }

                    ~FinallyBlock()
                    {
                        eo::log << eo::debug << "[M" << comm.rank() << "] Frees all the idle." << std::endl;

                        // frees all the idle workers
                        timerStat.start("master_wait_for_idles");
                        std::vector<int> idles = assignmentAlgo.idles();
                        for(unsigned int i = 0; i < idles.size(); ++i)
                        {
                            comm.send( idles[i], Channel::Commands, Message::Finish );
                        }
                        timerStat.stop("master_wait_for_idles");

                        eo::log << eo::debug << "[M" << comm.rank() << "] Waits for all responses." << std::endl;

                        // wait for all responses
                        timerStat.start("master_wait_for_all_responses");
                        while( assignmentAlgo.availableWorkers() != totalWorkers )
                        {
                            bmpi::status status = comm.probe( bmpi::any_source, eo::mpi::Channel::Messages );
                            int wrkRank = status.source();
                            that.handleResponse( wrkRank );
                            comm.send( wrkRank, Channel::Commands, Message::Finish );
                            assignmentAlgo.confirm( wrkRank );
                        }
                        timerStat.stop("master_wait_for_all_responses");

                        eo::log << eo::debug << "[M" << comm.rank() << "] Leaving master task." << std::endl;
                    }

                    protected:

                    int totalWorkers;
                    AssignmentAlgorithm& assignmentAlgo;
                    Job< JobData > & that;

                    bmpi::communicator & comm;
                };

                /**
                 * @brief Master part of the job.
                 *
                 * Launches the parallelized job algorithm : while there is something to do (! IsFinished ), get a
                 * worker who will be the assignee ; if no worker is available, wait for a response, handle it and reask
                 * for an assignee. Then send the command and the task.
                 * Once there is no more to do (IsFinished), indicate to all available workers that they're free, wait
                 * for all the responses and send termination messages (see also FinallyBlock).
                 */
                void master( )
                {
                    int totalWorkers = assignmentAlgo.availableWorkers();
                    eo::log << eo::debug << "[M" << comm.rank() << "] Have " << totalWorkers << " workers." << std::endl;

                    try {
                        FinallyBlock finally( totalWorkers, assignmentAlgo, *this );
                        while( ! isFinished() )
                        {
                            timerStat.start("master_wait_for_assignee");
                            int assignee = assignmentAlgo.get( );
                            while( assignee <= 0 )
                            {
                                eo::log << eo::debug << "[M" << comm.rank() << "] Waitin' for node..." << std::endl;

                                bmpi::status status = comm.probe( bmpi::any_source, eo::mpi::Channel::Messages );
                                int wrkRank = status.source();

                                eo::log << eo::debug << "[M" << comm.rank() << "] Node " << wrkRank << " just terminated." << std::endl;

                                handleResponse( wrkRank );
                                assignmentAlgo.confirm( wrkRank );
                                assignee = assignmentAlgo.get( );
                            }
                            timerStat.stop("master_wait_for_assignee");

                            eo::log << eo::debug << "[M" << comm.rank() << "] Assignee : " << assignee << std::endl;

                            timerStat.start("master_wait_for_send");
                            comm.send( assignee, Channel::Commands, Message::Continue );
                            sendTask( assignee );
                            timerStat.stop("master_wait_for_send");
                        }
                    } catch( const std::exception & e )
                    {
                        std::string s = e.what();
                        s.append( " in eoMpi loop");
                        throw std::runtime_error( s );
                    }
                }

                /**
                 * @brief Worker part of the algorithm.
                 *
                 * The algorithm is more much simpler: wait for an order; if it's termination message, leave. Otherwise,
                 * prepare to work.
                 */
                void worker( )
                {
                    int order;

                    timerStat.start("worker_wait_for_order");
                    comm.recv( masterRank, Channel::Commands, order );
                    timerStat.stop("worker_wait_for_order");

                    while( true )
                    {
                        eo::log << eo::debug << "[W" << comm.rank() << "] Waiting for an order..." << std::endl;

                        if ( order == workerStopCondition )
                        {
                            eo::log << eo::debug << "[W" << comm.rank() << "] Leaving worker task." << std::endl;
                            return;
                        } else if( order == Message::Continue )
                        {
                            eo::log << eo::debug << "[W" << comm.rank() << "] Processing task..." << std::endl;
                            processTask( );
                        }

                        timerStat.start("worker_wait_for_order");
                        comm.recv( masterRank, Channel::Commands, order );
                        timerStat.stop("worker_wait_for_order");
                    }
                }

            public:

                /**
             * @brief Launches the job algorithm, according to the role of the host (roles are deduced from the
                 * master rank indicated in the constructor).
                 */
                void run( )
                {
                    ( _isMaster ) ? master( ) : worker( );
                }

                /**
                 * @brief Returns true if the current host is the master, false otherwise.
                 */
                bool isMaster( )
                {
                    return _isMaster;
                }

            protected:

                AssignmentAlgorithm& assignmentAlgo;
                int masterRank;
                const int workerStopCondition;
                bmpi::communicator& comm;

                JobStore<JobData>& store;
                SendTaskFunction<JobData> & sendTask;
                HandleResponseFunction<JobData> & handleResponse;
                ProcessTaskFunction<JobData> & processTask;
                IsFinishedFunction<JobData> & isFinished;

                bool _isMaster;
        };

        /**
         * @brief Job that will be launched only once.
         *
         * As explained in eo::mpi documentation, jobs can happen either a well known amount of times or an unknown
         * amount of times. This class implements the general case when the job is launched a well known amount of
         * times. The job will be terminated on both sides (master and worker) once the master would have said it.
         *
         * It uses the message Message::Finish as the termination message.
         *
         * @ingroup MPI
         */
        template< class JobData >
        class OneShotJob : public Job< JobData >
        {
            public:
                OneShotJob( AssignmentAlgorithm& algo,
                            int masterRank,
                            JobStore<JobData> & store )
                    : Job<JobData>( algo, masterRank, Message::Finish, store )
                {
                    // empty
                }
        };

        /**
         * @brief Job that will be launched an unknown amount of times, in worker side.
         *
         * As explained in eo::mpi documentation, jobs can happen either a well known amount of times or an unknown
         * amount of times. This class implements the general case when the job is launched an unknown amount of times, for
         * instance in a while loop. The master will run many jobs (or the same job many times), but the workers will
         * launch it only once.
         *
         * It uses the message Message::Kill as the termination message. This message can be launched with an EmptyJob,
         * launched only by the master. If no Message::Kill is sent on the Channels::Commands, the worker will wait
         * forever, which will cause a deadlock.
         *
         * @ingroup MPI
         */
        template< class JobData >
        class MultiJob : public Job< JobData >
        {
            public:
                MultiJob ( AssignmentAlgorithm& algo,
                            int masterRank,
                            JobStore<JobData> & store )
                    : Job<JobData>( algo, masterRank, Message::Kill, store )
                {
                    // empty
                }
        };
    }

    /**
     * @}
     */
}
# endif // __EO_MPI_H__

