#include <smp>
#include <iostream>
#include <vector>

using namespace paradiseo::smp;

int main()
{
	int n;
	std::vector<unsigned> value;
	
	//Test of Complete Topology
	n=5;
	Topology<Complete> topo_comp;
	topo_comp.construct(n);
	
    
	std::vector<unsigned> neighbors = topo_comp.getIdNeighbors(1);
    
    value.clear();
	value.push_back(0);
	value.push_back(2);
	value.push_back(3);
	value.push_back(4);
	assert(neighbors == value);
	
	neighbors=topo_comp.getIdNeighbors(2);
    
    value.clear();
	value.push_back(0);
	value.push_back(1);
	value.push_back(3);
	value.push_back(4);
	assert(neighbors == value);
	    
    //Isolate an node
    topo_comp.isolateNode(2);
    neighbors=topo_comp.getIdNeighbors(2);
    
	assert(neighbors.empty());	
	
    neighbors=topo_comp.getIdNeighbors(3);
    
    value.clear();
	value.push_back(0);
	value.push_back(1);
	value.push_back(4);
	assert(neighbors == value);
	
    //Re-construct Topology with different number of nodes
    n=3;	
    topo_comp.construct(n);
    neighbors=topo_comp.getIdNeighbors(2);
    
    value.clear();
	value.push_back(0);
	value.push_back(1);
	assert(neighbors == value);	

    n=8;
    topo_comp.construct(n);
    neighbors = topo_comp.getIdNeighbors(3);
        
    value.clear();
	value.push_back(0);
	value.push_back(1);
	value.push_back(2);
	value.push_back(4);    
	value.push_back(5);
	value.push_back(6);
	value.push_back(7);
	assert(neighbors == value);

/////////////////////////////////////////////////////////////////////////	
	//Test of Star Topology
	n=4;
	Topology<Star> topo_star;
	topo_star.construct(n);

	neighbors=topo_star.getIdNeighbors(0);    

    value.clear();
	assert(neighbors == value);
    //---------------------------------------
	neighbors=topo_star.getIdNeighbors(2);
	    
    value.clear();
	value.push_back(0);
	assert(neighbors == value);
	//---------------------------------------
	topo_star.getBuilder().setCenter(2);
	topo_star.construct(n);

	neighbors=topo_star.getIdNeighbors(0);    
    
    value.clear();
    value.push_back(2);
	assert(neighbors == value);
    //---------------------------------------
	neighbors=topo_star.getIdNeighbors(2);
    value.clear();
	assert(neighbors == value);
	
	
/////////////////////////////////////////////////////////////////////////
	//Test of Ring Topology
	n=8;
	Topology<Ring> topo_ring;
	topo_ring.construct(n);

	neighbors=topo_ring.getIdNeighbors(4);
	   
    value.clear();
	value.push_back(5);
	assert(neighbors == value);
    //---------------------------------------
	neighbors=topo_ring.getIdNeighbors(7);
    
    value.clear();
	value.push_back(0);
	assert(neighbors == value);
	//---------------------------------------
	neighbors=topo_ring.getIdNeighbors(0);

    value.clear();
	value.push_back(1);
	assert(neighbors == value);

/////////////////////////////////////////////////////////////////////////
	//Test of Hypercubic Topology
	n=2;
	Topology<Hypercubic> topo_hyper;
	topo_hyper.construct(n);
    
	neighbors=topo_hyper.getIdNeighbors(0);
	   
    value.clear();
	value.push_back(1);
	assert(neighbors == value);
    //------------------------------------
    n=4;
   	topo_hyper.construct(n);
   	
   	neighbors=topo_hyper.getIdNeighbors(1);
   		   
    value.clear();
	value.push_back(0);
	value.push_back(3);
	assert(neighbors == value);
   	//-------------------------------------
   	n=8;
   	topo_hyper.construct(n);
   	
   	neighbors=topo_hyper.getIdNeighbors(5);
   		   
    value.clear();
    value.push_back(1);
	value.push_back(4);
	value.push_back(7);
	assert(neighbors == value);
	
	/////////////////////////////////////////////////////////////////////////
	//Test of Mesh Topology
	n=9;
	Topology<Mesh> topo_mesh;
	topo_mesh.construct(n);
	neighbors=topo_mesh.getIdNeighbors(0);
    
    value.clear();
	value.push_back(1);
	value.push_back(3);
	assert(neighbors == value);
	//-------------------------------------
	topo_mesh.getBuilder().setRatio(0.4);
	topo_mesh.construct(n);
	neighbors=topo_mesh.getIdNeighbors(5);

    value.clear();
	value.push_back(6);
	//assert(neighbors == value);	
    //--------------------------------------
	n=8;
	topo_mesh.construct(n);
	neighbors=topo_mesh.getIdNeighbors(0);
	
    value.clear();
	value.push_back(1);
	value.push_back(4);
	assert(neighbors == value);        	
}
