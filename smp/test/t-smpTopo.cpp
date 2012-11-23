#include <topology/topology.h>
#include <topology/complete.h>
#include <topology/star.h>
#include <topology/ring.h>
#include <iostream>
#include <vector>

using namespace paradiseo::smp;

int main()
{
	int n;
	
	//Test of Complete Topology
	n=5;
	Topology<Complete> topo_comp;
	topo_comp.construct(n);
	
    std::cout << std::endl << "---------------" << std::endl << "Test of Complete Topology (" << n <<" nodes) :"<<std::endl;
    
	std::vector<unsigned> neighbors=topo_comp.getIdNeighbors(1);
	std::cout << "neighbors of Node 1 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
			
	neighbors=topo_comp.getIdNeighbors(2);
	std::cout <<std::endl << "Neighbors of Node 2 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;

    //Re-construct Topology with different number of nodes
    n=3;	
    topo_comp.construct(n);
    neighbors=topo_comp.getIdNeighbors(2);
    std::cout <<"Changing number of nodes to "<< n <<" : "<<std::endl;
	std::cout <<std::endl << "Neighbors of Node 2 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
	    std::cout << " " << neighbors[i];
	std::cout << std::endl;
	
	
	//Test of Star Topology
	n=4;
	Topology<Star> topo_star;
	topo_star.construct(n);

    std::cout << std::endl << "---------------" << std::endl << "Test of Star Topology (" << n <<" nodes) :" << std::endl;   
    
	neighbors=topo_star.getIdNeighbors(0);
	std::cout <<std::endl << "Neighbors of Node 0 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;

	neighbors=topo_star.getIdNeighbors(1);
	std::cout <<std::endl << "Neighbors of Node 1 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;

	//Test of Ring Topology
	n=8;
	Topology<Ring> topo_ring;
	topo_ring.construct(n);

    std::cout << std::endl << "---------------" << std::endl << "Test of Ring Topology (" << n <<" nodes) :" << std::endl;    

	neighbors=topo_ring.getIdNeighbors(4);
	std::cout <<std::endl << "Neighbors of Node 4 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;

	neighbors=topo_ring.getIdNeighbors(7);
	std::cout <<std::endl << "Neighbors of Node 7 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;
	
	neighbors=topo_ring.getIdNeighbors(0);
	std::cout <<std::endl << "Neighbors of Node 0 : "<<std::endl;
	for (int i=0; i < neighbors.size(); i++)
		std::cout << " " << neighbors[i];
	std::cout << std::endl;    
}
