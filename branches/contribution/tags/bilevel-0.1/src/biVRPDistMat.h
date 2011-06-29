#ifndef BIVRPDISTMAT_H
#define BIVRPDISTMAT_H
#include <cstring>
#include <vector>
#include <set>
#include <iostream>
#include <utility>
#include <utils/eoRNG.h>
#include <DistMat.h>
class biVRPDistMat :public DistMat{
public:
	void load(const std::string);
	unsigned int numberOfVehicle();
	unsigned int numberOfDepot();
	unsigned int numberOfRetailer();
	unsigned int numberOfPlant();
	bool isConstantDemand();
	bool isConstantCapacity();
	double demand(unsigned int);
	double availability(unsigned int);

	double distance(unsigned int,unsigned int);
	double depotDistance(unsigned int,unsigned int);
	double plantDepotDistance(unsigned int,unsigned int);
	double maxLoad();
	double maxDuration();

	double charge(unsigned int);

	typedef std::pair<unsigned int,double> ty;
	virtual std::multiset<ty, compaPair> getOrder(unsigned int);
	std::multiset<ty, compaPair> nearDepot(unsigned int);
	std::multiset<ty, compaPair> getOrderPlant(unsigned int);
	double cbOfPlant(unsigned int);
	double ccOfPlant(unsigned int);
private: 
	unsigned int vehicleNumber;
	unsigned int depotNumber;
	unsigned int plantNumber;
	unsigned int retailerNumber;
	bool constantDemand;
	double loadMax;
	double durationMax;
	std::vector<double> retailerDemand;
	bool constantCapacity;
	std::vector<double> vehicleCapacity;
	std::vector<double> plantAvailability;
	unsigned int indexret(unsigned int _a,unsigned int _b);
	double computeDist(double,double,double,double);
	std::vector<double> cb;
	std::vector<double> cc;

	//vecteurs de distance
	std::vector<double> retailerDistance;
	std::vector<double> depotRetailDistance ;
	std::vector<double> depotPlantDistance ;

	
	//classement des distances
	std::vector<std::multiset < ty , compaPair  > > classement;
	std::vector<std::multiset < ty , compaPair  > > plantNearestDepot;
	std::vector<std::multiset < ty , compaPair  > > nearestDepot;

};


#endif

