#include <peo>
#include <es.h>
#include <moeo>

peoMoeoPopEval<FlowShop> & do_make_para_eval(eoParser& _parser, eoState& _state)
{
  std::string benchmarkFileName = _parser.getORcreateParam(std::string(), "BenchmarkFile", "Benchmark file name (benchmarks are available at www.lifl.fr/~liefooga/benchmarks)", 'B',"Representation", true).value();
  if (benchmarkFileName == "")
    {
      std::string stmp = "*** Missing name of the benchmark file\n";
      stmp += "    Type '-B=the_benchmark_file_name' or '--BenchmarkFile=the_benchmark_file_name'\n";
      stmp += "    Benchmarks files are available at www.lifl.fr/~liefooga/benchmarks";
      throw std::runtime_error(stmp.c_str());
    }
  FlowShopBenchmarkParser fParser(benchmarkFileName);
  unsigned int M = fParser.getM();
  unsigned int N = fParser.getN();
  std::vector< std::vector<unsigned int> > p = fParser.getP();
  std::vector<unsigned int> d = fParser.getD();

  FlowShopEval* plainEval = new FlowShopEval(M, N, p, d);
  peoMoeoPopEval<FlowShop>* eval = new peoMoeoPopEval<FlowShop> (* plainEval);
  _state.storeFunctor(eval);
  return *eval;
}
