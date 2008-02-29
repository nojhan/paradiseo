/*
* <peoWrapper.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Sebastien Cahon, Alexandru-Adrian Tantar, Clive Canape
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#ifndef __peoParaAlgorithm_h
#define __peoParaAlgorithm_h


#include "core/runner.h"
#include "core/peo_debug.h"

//! @class peoWrapper
//! @brief Specific class for wrapping
//! @see Runner
//! @version 1.1
//! @date december 2007
class peoWrapper : public Runner
  {

  public:

    //! @brief constructor
    //! @param AlgorithmType& externalAlgorithm
    template< typename AlgorithmType > peoWrapper( AlgorithmType& externalAlgorithm )
        : algorithm( new Algorithm< AlgorithmType, void >( externalAlgorithm ) )
    {}

    //! @brief constructor
    //! @param AlgorithmType& externalAlgorithm
    //! @param AlgorithmDataType& externalData
    template< typename AlgorithmType, typename AlgorithmDataType > peoWrapper( AlgorithmType& externalAlgorithm, AlgorithmDataType& externalData )
        : algorithm( new Algorithm< AlgorithmType, AlgorithmDataType >( externalAlgorithm, externalData ) )
    {}

    //! @brief constructor
    //! @param AlgorithmReturnType& (*externalAlgorithm)()
    template< typename AlgorithmReturnType > peoWrapper( AlgorithmReturnType& (*externalAlgorithm)() )
        : algorithm( new FunctionAlgorithm< AlgorithmReturnType, void >( externalAlgorithm ) )
    {}

    //! @brief constructor
    //! @param AlgorithmReturnType& (*externalAlgorithm)( AlgorithmDataType& )
    //! @param AlgorithmDataType& externalData
    template< typename AlgorithmReturnType, typename AlgorithmDataType > peoWrapper( AlgorithmReturnType& (*externalAlgorithm)( AlgorithmDataType& ), AlgorithmDataType& externalData )
        : algorithm( new FunctionAlgorithm< AlgorithmReturnType, AlgorithmDataType >( externalAlgorithm, externalData ) )
    {}

    //! @brief destructor
    ~peoWrapper()
    {
      delete algorithm;
    }

    //! @brief function run
    void run()
    {
      algorithm->operator()();
    }


  private:

    struct AbstractAlgorithm
      {

        // virtual destructor as we will be using inheritance and polymorphism
        virtual ~AbstractAlgorithm()
        { }

        // operator to be called for executing the algorithm
        virtual void operator()()
        { }
      };

  template< typename AlgorithmType, typename AlgorithmDataType > struct Algorithm : public AbstractAlgorithm
      {

        Algorithm( AlgorithmType& externalAlgorithm, AlgorithmDataType& externalData )
            : algorithm( externalAlgorithm ), algorithmData( externalData )
        {}

        virtual void operator()()
        {
          algorithm( algorithmData );
        }

        AlgorithmType& algorithm;
        AlgorithmDataType& algorithmData;
      };

  template< typename AlgorithmType > struct Algorithm< AlgorithmType, void >  : public AbstractAlgorithm
      {

        Algorithm( AlgorithmType& externalAlgorithm ) : algorithm( externalAlgorithm )
        {}

        virtual void operator()()
        {
          algorithm();
        }

        AlgorithmType& algorithm;
      };

  template< typename AlgorithmReturnType, typename AlgorithmDataType > struct FunctionAlgorithm : public AbstractAlgorithm
      {

        FunctionAlgorithm( AlgorithmReturnType (*externalAlgorithm)( AlgorithmDataType& ), AlgorithmDataType& externalData )
            : algorithm( externalAlgorithm ), algorithmData( externalData )
        {}

        virtual void operator()()
        {
          algorithm( algorithmData );
        }

        AlgorithmReturnType (*algorithm)( AlgorithmDataType& );
        AlgorithmDataType& algorithmData;
      };

  template< typename AlgorithmReturnType > struct FunctionAlgorithm< AlgorithmReturnType, void > : public AbstractAlgorithm
      {

        FunctionAlgorithm( AlgorithmReturnType (*externalAlgorithm)() )
            : algorithm( externalAlgorithm )
        {}

        virtual void operator()()
        {
          algorithm();
        }

        AlgorithmReturnType (*algorithm)();
      };

  private:
    //! @param AbstractAlgorithm* algorithm
    AbstractAlgorithm* algorithm;
  };


#endif
