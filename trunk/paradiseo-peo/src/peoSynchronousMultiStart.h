#ifndef __peoSynchronousMultiStart_h
#define __peoSynchronousMultiStart_h

#include <vector>

#include "core/service.h"
#include "core/messaging.h"


template < typename EntityType > class peoSynchronousMultiStart : public Service {

public:

	template < typename AlgorithmType > peoSynchronousMultiStart( AlgorithmType& externalAlgorithm ) { 

		singularAlgorithm = new Algorithm< AlgorithmType >( externalAlgorithm );
		algorithms.push_back( singularAlgorithm );

		aggregationFunction = new NoAggregationFunction();
	}

	template < typename AlgorithmType, typename AggregationFunctionType > peoSynchronousMultiStart( std::vector< AlgorithmType* >& externalAlgorithms, AggregationFunctionType& externalAggregationFunction ) {

		for ( unsigned int index = 0; index < externalAlgorithms; index++ ) {

			algorithms.push_back( new Algorithm< AlgorithmType >( *externalAlgorithms[ index ] ) );
		}

		aggregationFunction = new Algorithm< AggregationFunctionType >( externalAggregationFunction );
	}


	~peoSynchronousMultiStart() {

		for ( unsigned int index = 0; index < data.size(); index++ ) delete data[ index ];
		for ( unsigned int index = 0; index < algorithms.size(); index++ ) delete algorithms[ index ];

		delete aggregationFunction;
	}


	template < typename Type > void operator()( Type& externalData ) {

		for ( typename Type::iterator externalDataIterator = externalData.begin(); externalDataIterator != externalData.end(); externalDataIterator++ ) {

 			data.push_back( new DataType< EntityType >( *externalDataIterator ) );
		}
		
		functionIndex = dataIndex = idx = num_term = 0;
		requestResourceRequest( data.size() * algorithms.size() );
		stop();
	}


	template < typename Type > void operator()( const Type& externalDataBegin, const Type& externalDataEnd ) {

		for ( Type externalDataIterator = externalDataBegin; externalDataIterator != externalDataEnd; externalDataIterator++ ) {

 			data.push_back( new DataType< EntityType >( *externalDataIterator ) );
		}
		
		functionIndex = dataIndex = idx = num_term = 0;
		requestResourceRequest( data.size() * algorithms.size() );
		stop();
	}


	void packData();

	void unpackData();

	void execute();

	void packResult();

	void unpackResult();

	void notifySendingData();

	void notifySendingAllResourceRequests();


private:

	template < typename Type > struct DataType;

	struct AbstractDataType {

		virtual ~AbstractDataType() { }

		template < typename Type > operator Type& () {

			return ( dynamic_cast< DataType< Type >& >( *this ) ).data;
		}
	};

	template < typename Type > struct DataType : public AbstractDataType {

		DataType( Type& externalData ) : data( externalData ) { }

		Type& data;
	};

	struct AbstractAlgorithm {

		virtual ~AbstractAlgorithm() { }

		virtual void operator()( AbstractDataType& dataTypeInstance ) {}
	};

	template < typename AlgorithmType > struct Algorithm : public AbstractAlgorithm {

		Algorithm( AlgorithmType& externalAlgorithm ) : algorithm( externalAlgorithm ) { }

		void operator()( AbstractDataType& dataTypeInstance ) { algorithm( dataTypeInstance ); }

		AlgorithmType& algorithm;
	}; 



	struct AbstractAggregationAlgorithm {

		virtual ~AbstractAggregationAlgorithm() { }

		virtual void operator()( AbstractDataType& dataTypeInstanceA, AbstractDataType& dataTypeInstanceB ) {};
	};

	template < typename AggregationAlgorithmType > struct AggregationAlgorithm : public AbstractAggregationAlgorithm {

		AggregationAlgorithm( AggregationAlgorithmType& externalAggregationAlgorithm ) : aggregationAlgorithm( externalAggregationAlgorithm ) { }

		void operator()( AbstractDataType& dataTypeInstanceA, AbstractDataType& dataTypeInstanceB ) {

			aggregationAlgorithm( dataTypeInstanceA, dataTypeInstanceB );
		}

		AggregationAlgorithmType& aggregationAlgorithm;
	};

	struct NoAggregationFunction : public AbstractAggregationAlgorithm {

		void operator()( AbstractDataType& dataTypeInstanceA, AbstractDataType& dataTypeInstanceB ) {

			static_cast< EntityType& >( dataTypeInstanceA ) = static_cast< EntityType& >( dataTypeInstanceB );
		}
	};



	AbstractAlgorithm* singularAlgorithm;

	std::vector< AbstractAlgorithm* > algorithms;
	AbstractAggregationAlgorithm* aggregationFunction;


	EntityType entityTypeInstance;
	std::vector< AbstractDataType* > data;

	unsigned idx;
	unsigned num_term;
	unsigned dataIndex;
	unsigned functionIndex;
};


template < typename EntityType > void peoSynchronousMultiStart< EntityType >::packData() {

	::pack( functionIndex );
	::pack( idx );
	::pack( ( EntityType& ) *data[ idx++ ]  );

	// done with functionIndex for the entire data set - moving to another
	//  function/algorithm starting all over with the entire data set ( idx is set to 0 )
	if ( idx == data.size() ) {

		++functionIndex; idx = 0;
	}
}

template < typename EntityType > void peoSynchronousMultiStart< EntityType >::unpackData() {

	::unpack( functionIndex );
	::unpack( dataIndex );
 	::unpack( entityTypeInstance );
}

template < typename EntityType > void peoSynchronousMultiStart< EntityType >::execute() {

	// wrapping the unpacked data - the definition of an abstract algorithm imposes
	// that its internal function operator acts only on abstract data types
	AbstractDataType* entityWrapper = new DataType< EntityType >( entityTypeInstance );
	algorithms[ functionIndex ]->operator()( *entityWrapper );

	delete entityWrapper;
}

template < typename EntityType > void peoSynchronousMultiStart< EntityType >::packResult() {

	::pack( dataIndex );
	::pack( entityTypeInstance );
}

template < typename EntityType > void peoSynchronousMultiStart< EntityType >::unpackResult() {

	::unpack( dataIndex );
	::unpack( entityTypeInstance );

	// wrapping the unpacked data - the definition of an abstract algorithm imposes
	// that its internal function operator acts only on abstract data types
	AbstractDataType* entityWrapper = new DataType< EntityType >( entityTypeInstance );
	aggregationFunction->operator()( *data[ dataIndex ], *entityWrapper );
	delete entityWrapper;

	num_term++;

	if ( num_term == data.size() * algorithms.size() ) {

		getOwner()->setActive();
		resume();
	}
}

template < typename EntityType > void peoSynchronousMultiStart< EntityType >::notifySendingData() {

}

template < typename EntityType > void peoSynchronousMultiStart< EntityType >::notifySendingAllResourceRequests() {

	getOwner()->setPassive();
}


#endif
