// "peoParallelAlgorithmWrapper.h"

// (c) OPAC Team, LIFL, September 2007

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __peoParaAlgorithm_h
#define __peoParaAlgorithm_h


#include "core/runner.h"
#include "core/peo_debug.h"




class peoParallelAlgorithmWrapper : public Runner {

public:

	template< typename AlgorithmType > peoParallelAlgorithmWrapper( AlgorithmType& externalAlgorithm ) 
		: algorithm( new Algorithm< AlgorithmType, void >( externalAlgorithm ) ) {

	}

	template< typename AlgorithmType, typename AlgorithmDataType > peoParallelAlgorithmWrapper( AlgorithmType& externalAlgorithm, AlgorithmDataType& externalData ) 
		: algorithm( new Algorithm< AlgorithmType, AlgorithmDataType >( externalAlgorithm, externalData ) ) {

	}

	~peoParallelAlgorithmWrapper() {

		delete algorithm;
	}

	void run() { algorithm->operator()(); }


private:

        struct AbstractAlgorithm {

		// virtual destructor as we will be using inheritance and polymorphism
		virtual ~AbstractAlgorithm() { }

		// operator to be called for executing the algorithm
		virtual void operator()() { } 
        };


        template< typename AlgorithmType, typename AlgorithmDataType > struct Algorithm : public AbstractAlgorithm {

		Algorithm( AlgorithmType& externalAlgorithm, AlgorithmDataType& externalData ) 
			: algorithm( externalAlgorithm ), algorithmData( externalData ) {

		}

		virtual void operator()() { algorithm( algorithmData ); } 

		AlgorithmType& algorithm;
		AlgorithmDataType& algorithmData;
        };


        template< typename AlgorithmType > struct Algorithm< AlgorithmType, void >  : public AbstractAlgorithm {

		Algorithm( AlgorithmType& externalAlgorithm ) : algorithm( externalAlgorithm ) {

		}

		virtual void operator()() { algorithm(); } 

		AlgorithmType& algorithm;
        };


private:

	AbstractAlgorithm* algorithm;
};


#endif
