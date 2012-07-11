# ifndef __TIMER_H__
# define __TIMER_H__

# include <sys/time.h>
# include <sys/resource.h>

# include <vector>
# include <map>

# include "utils/eoParallel.h"

# ifdef WITH_MPI
# include <boost/serialization/access.hpp>
# include <boost/serialization/vector.hpp>
# include <boost/serialization/map.hpp>
# endif

// TODO TODOB commenter
class eoTimer
{
    public:

        eoTimer()
        {
            restart();
        }

        void restart()
        {
            uuremainder = 0;
            usremainder = 0;
            wc_start = time(NULL);
            getrusage( RUSAGE_SELF, &_start );
        }

        long int usertime()
        {
            struct rusage _now;
            getrusage( RUSAGE_SELF, &_now );
            long int result = _now.ru_utime.tv_sec - _start.ru_utime.tv_sec;
            if( _now.ru_utime.tv_sec == _start.ru_utime.tv_sec )
            {
                uuremainder += _now.ru_utime.tv_usec - _start.ru_utime.tv_usec;
                if( uuremainder > 1000000)
                {
                    ++result;
                    uuremainder = 0;
                }
            }
            return result;
        }

        long int systime()
        {
            struct rusage _now;
            getrusage( RUSAGE_SELF, &_now );
            long int result = _now.ru_stime.tv_sec - _start.ru_stime.tv_sec;
            if( _now.ru_stime.tv_sec == _start.ru_stime.tv_sec )
            {
                usremainder += _now.ru_stime.tv_usec - _start.ru_stime.tv_usec;
                if( usremainder > 1000000)
                {
                    ++result;
                    usremainder = 0;
                }
            }
            return result;
        }

        double wallclock()
        {
            return std::difftime( std::time(NULL) , wc_start );
        }

    protected:
        struct rusage _start;
        long int uuremainder;
        long int usremainder;
        time_t wc_start;
};

class eoTimerStat
{
    public:

        struct Stat
        {
            std::vector<long int> utime;
            std::vector<long int> stime;
            std::vector<double> wtime;
#ifdef WITH_MPI
            // Gives access to boost serialization
            friend class boost::serialization::access;

            /**
             * Serializes the statistique in a boost archive (useful for boost::mpi)
             */
            template <class Archive>
            void serialize( Archive & ar, const unsigned int version )
            {
                ar & utime & stime & wtime;
                (void) version; // avoid compilation warning
            }
# endif
        };

#ifdef WITH_MPI
            // Gives access to boost serialization
            friend class boost::serialization::access;

            /**
             * Serializes the map of statistics in a boost archive (useful for boost::mpi)
             */
            template <class Archive>
            void serialize( Archive & ar, const unsigned int version )
            {
                ar & _stats;
                (void) version; // avoid compilation warning
            }
# endif

        void start( const std::string & key )
        {
            if( eo::parallel.doMeasure() )
            {
                _timers[ key ].restart();
            }
        }

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

        std::map< std::string, Stat > stats()
        {
            return _stats;
        }

    protected:
        std::map< std::string, Stat > _stats;
        std::map< std::string, eoTimer > _timers;
};

# endif // __TIMER_H__

