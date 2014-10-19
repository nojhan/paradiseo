#include <paradiseo/smp/topology/customBooleanTopology.h>
#include <paradiseo/smp/topology/customStochasticTopology.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <cmath>

using namespace paradiseo::smp;

int main()
{

    std::vector<unsigned> value;

    //TEST OF CUSTOM BOOLEAN TOPOLOGY
	CustomBooleanTopology topo_bool("data-topo-bool");
    std::vector<unsigned> neighbors = topo_bool.getIdNeighbors(0);
    
    value.clear();
	value.push_back(1);
	assert(neighbors == value);
///////////////////////////////////////////////////////	
	neighbors = topo_bool.getIdNeighbors(1);
    
    value.clear();
	value.push_back(0);
	value.push_back(2);
	value.push_back(3);
	assert(neighbors == value);
//////////////////////////////////////////////////////
	neighbors = topo_bool.getIdNeighbors(2);
		
    value.clear();
	value.push_back(1);
	value.push_back(3);
	assert(neighbors == value);
//////////////////////////////////////////////////////
	neighbors = topo_bool.getIdNeighbors(3);
    
    value.clear();
	value.push_back(0);
	value.push_back(2);
	assert(neighbors == value);
	
	
    //TEST OF CUSTOM STOCHASTIC TOPOLOGY
    CustomStochasticTopology topo_stoch("data-topo-stoch");
    std::vector<std::vector<double>> matrix;
    matrix.resize(3);
    for(auto& line : matrix)
        line.resize(3);
        
    //Computation of the mean probability    
    int it_nb = 1000;
    for(int i = 0 ; i < it_nb ; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            neighbors = topo_stoch.getIdNeighbors(j);
            for(auto& node : neighbors)
            {
                matrix[j][node]++;
            }
        }
    }
    
    for(auto& line : matrix)
    {
        for(auto& edge : line)
            edge = edge/it_nb;
    }
    
    //Reading the actual matrix
    std::ifstream f("data-topo-stoch");
    std::vector<std::vector<double>> _matrix;
    if (f)
    {
        double temp;
        double isNeighbor;
        std::string line;
        std::vector<double> lineVector;
        
        while(getline(f, line))
        {
            lineVector.clear();
            
            //line contains a line of text from the file
            std::istringstream tokenizer(line);
            std::string token;
            
            while(tokenizer >> temp >> std::skipws)
            {
                //white spaces are skipped, and the integer is converted to boolean, to be stored
                
                if(temp<0)
                    isNeighbor = 0;
                else if(temp>1)
                    isNeighbor = 1;
                else
                    isNeighbor = (double) temp;
                lineVector.push_back(isNeighbor);
            }
            
            if(!lineVector.empty())
                _matrix.push_back(lineVector);
        }

        f.close () ;
    }
    
    //Comparison to the actual matrix : _matrix
    for(unsigned i = 0; i < matrix.size(); i++)
        for(unsigned j = 0; j < matrix.size(); j++)
            assert(std::abs(_matrix[i][j] - matrix[i][j]) < .05);
}
