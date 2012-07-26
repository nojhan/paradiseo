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
# ifndef __EO_TIMER_H__
# define __EO_TIMER_H__

# include <sys/time.h> // time()
# include <sys/resource.h> // rusage()

# include <vector> // std::vector
# include <map> // std::map

# include "utils/eoParallel.h" // eo::parallel

# include "serial/eoSerial.h" // eo::Persistent

/**
 * @brief Timer allowing to measure time between a start point and a stop point.
 *
 * This timer allows the user to measure user time, system time and wallclock time
 * between two points. Basically, user time is time spent in developer code ; system
 * time is time spent during the IO wait and system calls ; wallclock is the difference
 * of time we could observe by measuring time with a watch.
 *
 * @ingroup Utilities
 */
class eoTimer
{
    public:

        /**
         * @brief Default ctor. Begins all the timers.
         */
        eoTimer()
        {
            uuremainder = 0;
            usremainder = 0;
            restart();
        }

        /**
         * @brief Restarts all the timers and launch the measure.
         */
        void restart()
        {
            wc_start = time(NULL);
            getrusage( RUSAGE_SELF, &_start );
        }

        /**
         * @brief Measures the user time spent since the last restart().
         *
         * If the number of elapsed seconds is zero, the spent milliseconds are
         * added to a remainder. If the remainder exceeds one second, it is
         * added to the number of elapsed seconds.
         *
         * @return Number of seconds elapsed in user space.
         */
        long int usertime()
        {
            struct rusage _now;
            getrusage( RUSAGE_SELF, &_now );

            long int result = _now.ru_utime.tv_sec - _start.ru_utime.tv_sec;
            long int remainder = _now.ru_utime.tv_usec - _start.ru_utime.tv_usec;
            if( remainder >= 0 )
            {
                uuremainder += remainder;
            } else
            {
                uuremainder += ( 1000000 - remainder );
                --result;
            }

            if( uuremainder >= 1000000 )
            {
                uuremainder -= 1000000;
                ++result;
            }
            return result;
        }

        /**
         * @brief Measures the system time spent since the last restart().
         *
         * If the number of elapsed seconds is zero, the spent milliseconds are
         * added to a remainder. If the remainder exceeds one second, it is
         * added to the number of elapsed seconds.
         *
         * @return Number of seconds elapsed in system (kernel) space.
         */
        long int systime()
        {
            struct rusage _now;
            getrusage( RUSAGE_SELF, &_now );

            long int result = _now.ru_stime.tv_sec - _start.ru_stime.tv_sec;
            long int remainder = _now.ru_stime.tv_usec - _start.ru_stime.tv_usec;
            if( remainder >= 0 )
            {
                usremainder += remainder;
            } else
            {
                usremainder += ( 1000000 - remainder );
                --result;
            }

            if( usremainder >= 1000000 )
            {
                usremainder -= 1000000;
                ++result;
            }
            return result;
        }

        /**
         * @brief Measures the wallclock time spent since the last restart().
         *
         * @return Number of seconds elapsed, as a double.
         */
        double wallclock()
        {
            return std::difftime( std::time(NULL) , wc_start );
        }

    protected:
        // Structure used to measure user and system time.
        struct rusage _start;
        // Remainder (in milliseconds) for user time.
        long int uuremainder;
        // Remainder (in milliseconds) for system time.
        long int usremainder;
        // Structure used to measure wallclock time.
        time_t wc_start;
};

/**
 * @brief Registers a group of statistics, each statistic corresponding to user, system and wallclock times distribution.
 *
 * This class helps the user to measure time in different parts of an application. A name is associated to a statistic,
 * on each call to start() and stop() for this name, a new number is added to the statistic, for each of the three
 * measured times.
 *
 * The statistics are only registered if the option "--parallelized-do-measure" is set to true, which can be checked
 * thanks to global object eo::parallel.
 *
 * This shows how the eoTimerStat can be used :
 * @code
 * eoTimerStat timerStat;
 * timerStat.start("first_point");
 * for( int i = 0; i < 1000; ++i )
 * {
 *   timerStat.start("single_computation");
 *   single_computation( i );
 *   timerStat.stop("single_computation");
 * }
 * // After this loop, timerStat contains a statistic of key "single_computation" which contains 1000 measures for
 * // each type of time.
 * timerStat.stop("first_point");
 * // After this line, timerStat contains another statistic of key "first_point" which counted the duration of the
 * // whole loop.
 *
 * int singleComputationUsertimeMean = 0;
 * for( int i = 0; i < 1000; ++i )
 * {
 *      singleComputationUsertimeMean += timerStat.stats()["single_computation"].utime[i];
 * }
 * std::cout << "Mean of user time spent in single computation: " << singleComputationUsertimeMean / 1000. << std::endl;
 * @endcode
 *
 * When using MPI, these statistics can be readily be serialized, so as to be sent over a network, for instance.
 *
 * Implementation details: this eoTimerStat is in fact a map of strings (key) / Stat (value). Stat is an internal
 * structure directly defined in the class, which contains three vectors modeling the distributions of the different
 * types of elapsed times. Another map of strings (key) / eoTimer (value) allows to effectively retrieve the different
 * times. The struct Stat will be exposed to client, which will use its members ; however,
 * the client doesn't have anything to do directly with the timer, that's why the two maps are splitted.
 *
 * @ingroup Utilities
 */
class eoTimerStat
# ifdef WITH_MPI
    : public eoserial::Persistent
# endif
{
    public:

        /**
         * @brief Statistic related to a key (name).
         *
         * This structure is the value in the map saved in the eoTimerStat. It contains the statistic bound to a key,
         * which are the user time distribution, the system time distribution and the wallclock time distribution, as
         * std::vector s.
         *
         * It can readily be serialized with boost when compiling with mpi.
         */
        struct Stat
# ifdef WITH_MPI
            : public eoserial::Persistent
# endif
        {
            std::vector<long int> utime;
            std::vector<long int> stime;
            std::vector<double> wtime;
#ifdef WITH_MPI
            void unpack( const eoserial::Object* obj )
            {
                utime.clear();
                static_cast< eoserial::Array* >(obj->find("utime")->second)
                    ->deserialize< std::vector<long int>, eoserial::Array::UnpackAlgorithm >( utime );

                stime.clear();
                static_cast< eoserial::Array* >(obj->find("stime")->second)
                    ->deserialize< std::vector<long int>, eoserial::Array::UnpackAlgorithm >( stime );

                wtime.clear();
                static_cast< eoserial::Array* >(obj->find("wtime")->second)
                    ->deserialize< std::vector<double>, eoserial::Array::UnpackAlgorithm >( wtime );
            }

            eoserial::Object* pack( void ) const
            {
                eoserial::Object* obj = new eoserial::Object;
                obj->add("utime", eoserial::makeArray< std::vector<long int>, eoserial::MakeAlgorithm >( utime ) );
                obj->add("stime", eoserial::makeArray< std::vector<long int>, eoserial::MakeAlgorithm >( stime ) );
                obj->add("wtime", eoserial::makeArray< std::vector<double>, eoserial::MakeAlgorithm >( wtime ) );
                return obj;
            }
# endif
        };

#ifdef WITH_MPI
        void unpack( const eoserial::Object* obj )
        {
            _stats.clear();
            for( eoserial::Object::const_iterator it = obj->begin(), final = obj->end();
                    it != final;
                    ++it)
            {
                eoserial::unpackObject( *obj, it->first, _stats[ it->first ] );
            }
        }

        eoserial::Object* pack( void ) const
        {
            eoserial::Object* obj = new eoserial::Object;
            for( std::map<std::string, Stat >::const_iterator it = _stats.begin(), final = _stats.end();
                    it != final;
                    ++it)
            {
                obj->add( it->first, it->second.pack() );
            }
            return obj;
        }
# endif

        /**
         * @brief Starts a new measure for the given key.
         *
         * This is only performed if parallel.doMeasure() is true, which is equivalent to the fact that
         * parser found "--parallel-do-measure=1" in command line args.
         *
         * @param key The key of the statistic.
         */
        void start( const std::string & key )
        {
            if( eo::parallel.doMeasure() )
            {
                _timers[ key ].restart();
            }
        }

        /**
         * @brief Stops the mesure for the given key and saves the elapsed times.
         *
         * Must follow a call of start with the same key.
         *
         * This is only performed if parallel.doMeasure() is true, which is equivalent to the fact that
         * parser found "--parallel-do-measure=1" in command line args.
         *
         * @param key The key of the statistic.
         */
        void stop( const std::string& key )
        {
            if( eo::parallel.doMeasure() )
            {
                Stat & s = _stats[ key ];
                eoTimer & t = _timers[ key ];
                s.utime.push_back( t.usertime() );
                s.stime.push_back( t.systime() );
                s.wtime.push_back( t.wallclock() );
            }
        }

        /**
         * @brief Getter for the statistics map.
         */
        std::map< std::string, Stat >& stats()
        {
            return _stats;
        }

    protected:
        // Statistics map: links a key (string) to a statistic.
        std::map< std::string, Stat > _stats;
        // Timers map: links a key to its timer.
        std::map< std::string, eoTimer > _timers;
};

# endif // __TIMER_H__

