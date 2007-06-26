// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopBenchmarkParser.cpp
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#include <stdexcept>
#include <FlowShopBenchmarkParser.h>

FlowShopBenchmarkParser::FlowShopBenchmarkParser(const std::string _benchmarkFileName)
{
    init(_benchmarkFileName);
}


const unsigned int FlowShopBenchmarkParser::getM()
{
    return M;
}


const unsigned int FlowShopBenchmarkParser::getN()
{
    return N;
}


const std::vector< std::vector<unsigned int> > FlowShopBenchmarkParser::getP()
{
    return p;
}


const std::vector<unsigned int> FlowShopBenchmarkParser::getD()
{
    return d;
}


void FlowShopBenchmarkParser::printOn(std::ostream & _os) const
{
    _os << "M=" << M << " N=" << N << std::endl;
    _os << "*** processing times" << std::endl;
    for (unsigned int i=0; i<M; i++) {
        for (unsigned int j=0; j<N; j++) {
            _os << p[i][j] << " ";
        }
        _os << std::endl;
    }
    _os << "*** due-dates" << std::endl;
    for (unsigned int j=0; j<N; j++) {
        _os << d[j] << " ";
    }
    _os << std::endl << std::endl;
}


void FlowShopBenchmarkParser::init(const std::string _benchmarkFileName)
{
    std::string buffer;
    std::string::size_type start, end;
    std::ifstream inputFile(_benchmarkFileName.data(), std::ios::in);
    // opening of the benchmark file
    if (! inputFile)
    	throw std::runtime_error("*** ERROR : Unable to open the benchmark file");
    // number of jobs (N)
    getline(inputFile, buffer, '\n');
    N = atoi(buffer.data());
    // number of machines M
    getline(inputFile, buffer, '\n');
    M = atoi(buffer.data());
    // initial and current seeds (not used)
    getline(inputFile, buffer, '\n');
    // processing times and due-dates
    p = std::vector< std::vector<unsigned int> > (M,N);
    d = std::vector<unsigned int> (N);
    // for each job...
    for (unsigned int j=0 ; j<N ; j++) {
        // index of the job (<=> j)
        getline(inputFile, buffer, '\n');
        // due-date of the job j
        getline(inputFile, buffer, '\n');
        d[j] = atoi(buffer.data());
        // processing times of the job j on each machine
        getline(inputFile, buffer, '\n');
        start = buffer.find_first_not_of(" ");
        for (unsigned int i=0 ; i<M ; i++) {
            end = buffer.find_first_of(" ", start);
            p[i][j] = atoi(buffer.substr(start, end-start).data());
            start = buffer.find_first_not_of(" ", end);
        }
    }
    // closing of the input file
    inputFile.close();
}
