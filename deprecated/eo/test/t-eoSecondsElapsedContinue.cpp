#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <time.h>
#include <eoSecondsElapsedContinue.h>
#include <eoPop.h>
#include <EO.h>

class Dummy : public EO<double> {};

int main() {

    eoPop<Dummy> pop;

    eoSecondsElapsedContinue<Dummy> cnt(1);

	time_t start_time = time(0);

    while (cnt(pop)) {}

    time_t end_time = time(0);

    int diff = end_time = start_time;

    if (diff < 1) return 1;

    return 0;

}
