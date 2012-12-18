#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <sstream>

#include <utils/eoState.h>
#include <utils/eoUpdater.h>

using namespace std;

void eoTimedStateSaver::operator()(void)
{
    time_t now = time(0);

    if (now >= last_time + interval)
    {
        last_time = now;
        ostringstream os;
        os << prefix << (now - first_time) << '.' << extension;
        state.save(os.str());
    }
}

void eoCountedStateSaver::doItNow(void)
{
        ostringstream os;
        os << prefix << counter << '.' << extension;
        state.save(os.str());
}

void eoCountedStateSaver::operator()(void)
{
    if (++counter % interval == 0)
      doItNow();
}

void eoCountedStateSaver::lastCall(void)
{
    if (saveOnLastCall)
      doItNow();
}
