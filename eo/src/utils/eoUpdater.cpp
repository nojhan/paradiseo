#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif


#include <strstream>

#include <utils/eoState.h>
#include <utils/eoUpdater.h>

using namespace std;

void eoTimedStateSaver::operator()(void)
{
    time_t now = time(0);

    if (now >= last_time + interval)
    {
        last_time = now;

        ostrstream os;
        os << prefix << (now - first_time) << '.' << extension << ends;
        state.save(os.str());
    }
}

void eoCountedStateSaver::doItNow(void)
{
  ostrstream os;
  os << prefix << counter << '.' << extension << ends;
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


