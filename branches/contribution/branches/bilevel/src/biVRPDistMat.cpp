#include <biVRPDistMat.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <utils/eoRNG.h>

void biVRPDistMat::load(const std::string _fname){
	std::string buf;
	std::ifstream file(_fname.data());
	if (file.is_open()){
		int type;
		file >> type;
		if (type!=8){
			std::cout<<"pas un bimdvrp"<<std::endl;
			exit(1);
		}
		file>>vehicleNumber;
		file>>retailerNumber;
		file>>depotNumber;
		file>>plantNumber;
		file>>durationMax;
		file>>loadMax;
		vehicleNumber*=depotNumber;
		retailerDistance.resize(retailerNumber*(retailerNumber+1)/2);
		depotRetailDistance.resize(retailerNumber*depotNumber);
		depotPlantDistance.resize(plantNumber*depotNumber);
		nearestDepot.resize(retailerNumber);
		plantNearestDepot.resize(plantNumber);
		classement.resize(retailerNumber);

		for(unsigned int i=1;i<depotNumber;i++){
			unsigned int dur,load;
			file>>dur;
			file>>load;
			if (dur!=durationMax || load!=loadMax)
				std::cout<<"ne gere pas les durées max ou les charge max multiples("<<load<<"!="<<loadMax<<std::endl;
		}

		std::vector< std::vector <double> > coords;
		for (unsigned int i=0;i<retailerNumber+depotNumber+plantNumber;i++){
			std::vector<double>coord;
			double x,y,dump,demand;
			double n;
			double neocb;
			file>>dump;
			file>>x;
			file>>y;
			coord.push_back(x);
			coord.push_back(y);
			coords.push_back(coord);
			file>>dump;
			file>>demand;
			file>>neocb;
			file>>n;
			if (i<retailerNumber) retailerDemand.push_back(demand);
			if (i>=retailerNumber+depotNumber && i<retailerNumber+depotNumber+plantNumber) {
				plantAvailability.push_back(demand);
				double neocc=n;
				cb.push_back(neocb);
				cc.push_back(neocc);
			}
			else for (unsigned int j=0;j<n;j++) file>>dump;

		
		
		}
		for (unsigned int i=0;i<retailerNumber;i++){
			for(unsigned int j=i+1;j<retailerNumber;j++){
				unsigned int idx=indexret(i,j);
				retailerDistance[idx]=computeDist(coords[i][0],coords[i][1], coords[j][0],coords[j][1]);
				classement[i].insert(std::make_pair(j,retailerDistance[idx]));
				classement[j].insert(std::make_pair(i,retailerDistance[idx]));
			}
		}
		for (unsigned int i=0;i<depotNumber;i++){
			for(unsigned int j=0;j<retailerNumber;j++){
				unsigned int idx=i*numberOfRetailer()+ j;
				depotRetailDistance[idx]=computeDist(
						coords[numberOfRetailer()+i][0],
						coords[numberOfRetailer()+i][1],
						coords[j][0],
						coords[j][1]);
				nearestDepot[j].insert(std::make_pair(i,depotRetailDistance[idx]));
			}
			for(unsigned int j=0;j<plantNumber;j++){
				unsigned int idx=i*numberOfPlant()+j;
				depotPlantDistance[idx]=computeDist(
						coords[numberOfRetailer()+numberOfPlant()+i][0],
						coords[numberOfRetailer()+numberOfPlant()+i][1],
						coords[j][0],
						coords[j][1]);
				plantNearestDepot[j].insert(std::make_pair(i,depotPlantDistance[idx]));
			}
		}



	}
}
double biVRPDistMat::cbOfPlant(unsigned int i){
	return cb[i];
}
double biVRPDistMat::ccOfPlant(unsigned int i){
	return cc[i];
}
unsigned int biVRPDistMat::numberOfVehicle(){
	return vehicleNumber;
}
unsigned  int biVRPDistMat::numberOfDepot(){
	return depotNumber;
}
unsigned int biVRPDistMat::numberOfRetailer(){
	return retailerNumber;
}
unsigned int biVRPDistMat::numberOfPlant(){
	return plantNumber;
}
bool biVRPDistMat::isConstantDemand(){
	return constantDemand;
}
bool biVRPDistMat::isConstantCapacity(){
	return constantCapacity;
}
double biVRPDistMat::demand(unsigned int retailer){
	if (retailer<numberOfRetailer()){
		return retailerDemand[retailer];
	}else{ 
		return -1;
	}
}
double biVRPDistMat::availability( unsigned int plant){
	if (plant<numberOfPlant()){
		return plantAvailability[plant];
	}else{ 
		return -1;
	}

}
double biVRPDistMat::distance(unsigned int retailerA,unsigned int retailerB){
	if (retailerA < numberOfRetailer() && retailerB < numberOfRetailer()){
		return retailerDistance[indexret(retailerA,retailerB)];
	}else{
		return -1;
	}
}

double biVRPDistMat::depotDistance(unsigned int depot,unsigned int retailer){
	if(depot<numberOfDepot() && retailer < numberOfRetailer()){
		int idx=depot*numberOfRetailer()+ retailer;
		return depotRetailDistance[idx];
	}else{
		return -1;
	}
}

double biVRPDistMat::plantDepotDistance(unsigned int depot,unsigned int plant){
	if(depot<numberOfDepot() && plant < numberOfPlant()){
		int idx=depot*numberOfPlant()+ plant;
		return depotPlantDistance[idx];
	}else{
		return -1;
	}
}
double biVRPDistMat::maxLoad(){
	return loadMax;
}


double biVRPDistMat::maxDuration(){
	return 0;
}
std::multiset<biVRPDistMat::ty, biVRPDistMat::compaPair> biVRPDistMat::getOrder(unsigned int retail){
	return classement[retail];
}
std::multiset<biVRPDistMat::ty, biVRPDistMat::compaPair> biVRPDistMat::getOrderPlant(unsigned int plant){
	return plantNearestDepot[plant];
}
std::multiset<biVRPDistMat::ty, biVRPDistMat::compaPair> biVRPDistMat::nearDepot(unsigned int i){
	return nearestDepot[i];
}
unsigned int biVRPDistMat::indexret(unsigned int _a,unsigned int _b){
	int a,b;
	if (_b<_a){
		a=_a;
		b=_b;
	}else{
		a=_b;
		b=_a;
	}
	return b*numberOfRetailer()+a - (b*(b+1)/2);
}


double biVRPDistMat::computeDist(double xa,double ya,double xb,double yb){
	double dist;
	dist=sqrt(pow(xa-xb,2)+pow(ya-yb,2));
	return dist;
}

