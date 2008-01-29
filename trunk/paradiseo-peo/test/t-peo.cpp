
#include <peo.h>

void EAParaEval ()
{
  char *tmp="mpiexec -n 4 ./t-EAParaEval @param ";
  system(tmp);
}

int main (int __argc, char *__argv[])
{
  EAParaEval();
}
