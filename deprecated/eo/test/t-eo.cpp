#include <eo>

typedef EO<float> Chrom;

int main()
{
  Chrom chrom1, chrom2;

  // EO objects can be printed with stream operators
  std::cout << "chrom1 = " << chrom1 << std::endl
	    << "chrom2 = " << chrom2 << std::endl;

  return 0;
}
