#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#ifdef HAVE_SSTREAM
#include <sstream>
#else
#include <strstream>
#endif

#include <utils/eoState.h>
#include <utils/eoUpdater.h>

using namespace std;

void eoTimedStateSaver::operator()(void)
{
    time_t now = time(0);

    if (now >= last_time + interval)
    {
        last_time = now;
        
#ifdef HAVE_SSTREAM
	ostringstream os;
#else
	ostrstream os;
#endif
        os << prefix << (now - first_time) << '.' << extension << ends;
        state.save(os.str());
    }
}

void eoCountedStateSaver::doItNow(void)
{
#ifdef HAVE_SSTREAM
	ostringstream os;
#else
	ostrstream os;
#endif
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


