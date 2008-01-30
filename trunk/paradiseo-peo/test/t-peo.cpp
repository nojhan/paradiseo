#include <peo.h>

int main (int __argc, char *__argv[])
{
	system("mpdboot");
	system("mpiexec -n 4 ./t-ParallelEval @param ");
	system("mpiexec -n 4 ./t-ParallelTransform @param ");
	system("mpiexec -n 4 ./t-MultiStart @param ");
	system("mpdallexit");
}
