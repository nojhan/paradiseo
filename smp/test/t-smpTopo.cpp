#include <topology/topology.h>
#include <topology/complete.h>
#include <topology/star.h>
#include <topology/ring.h>
#include <iostream>
#include <vector>

using namespace paradiseo::smp;

int main()
{
	//Test of Complete Topology
	Topology<Complete> topo_comp;
	topo_comp.construct(5);
	
    std::cout << std::endl << "---------------" << std::endl << "Test of Complete Topology" << std::endl;
    
	std::vector<unsigned> neighbors=topo_comp.getIdNeighbors(1);
	std::cout << "neighbors of Island 1 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
			
	neighbors=topo_comp.getIdNeighbors(2);
	std::cout <<std::endl << "Neighbors of Island 2 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;
	
	//Test of Star Topology
	Topology<Star> topo_star;
	topo_star.construct(4);

    std::cout << std::endl << "---------------" << std::endl << "Test of Star Topology" << std::endl;   
    
	neighbors=topo_star.getIdNeighbors(0);
	std::cout <<std::endl << "Neighbors of Island 0 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;

	neighbors=topo_star.getIdNeighbors(1);
	std::cout <<std::endl << "Neighbors of Island 1 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;

	//Test of Ring Topology
	Topology<Ring> topo_ring;
	topo_ring.construct(8);

    std::cout << std::endl << "---------------" << std::endl << "Test of Ring Topology" << std::endl;    

	neighbors=topo_ring.getIdNeighbors(4);
	std::cout <<std::endl << "Neighbors of Island 4 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;

	neighbors=topo_ring.getIdNeighbors(7);
	std::cout <<std::endl << "Neighbors of Island 7 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;
	
	neighbors=topo_ring.getIdNeighbors(0);
	std::cout <<std::endl << "Neighbors of Island 0 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;    
}
