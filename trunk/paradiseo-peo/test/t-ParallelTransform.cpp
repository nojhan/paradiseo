#include <peo.h>

int main (int __argc, char *__argv[])
{
	system("mpdboot");
	system("mpiexec -n 4 ./t-ParallelTransformLib @param ");
	system("mpdallexit");
}
