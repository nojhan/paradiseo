
######################################################################################
### 1) Definitions
######################################################################################

SET(FLOWSHOP_SRC_DIR ${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop)
SET(FLOWSHOP_BINARY_DIR ${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop)

######################################################################################

######################################################################################
### 2) How to manage copy of benchs ?
######################################################################################

IF(ENABLE_CMAKE_COPY)
	SET(BENCH_LIST
		020_05_01.txt
		020_05_02.txt
		020_10_01.txt
		020_10_02.txt
		020_20_01.txt
		050_05_01.txt
		050_10_01.txt
		050_20_01.txt
		100_05_01.txt
		100_10_01.txt
		100_20_01.txt
		200_10_01.txt	
		README	
	)
	
	FOREACH (bench ${BENCH_LIST})
			EXECUTE_PROCESS(
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
	      			${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/${bench}
	    			${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/${bench}
	    	)
	ENDFOREACH (bench)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson1/Sch1.param
			${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson1/Sch1.param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson2/FlowShopEA.param
			${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson2/FlowShopEA.param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson3/FlowShopEA2.param
			${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson3/FlowShopEA2.param
	)
ENDIF(ENABLE_CMAKE_COPY)

######################################################################################
### 3) Where must cmake go now ?
######################################################################################

#SUBDIRS(examples Lesson1 Lesson2 Lesson3)
ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(Lesson1)
ADD_SUBDIRECTORY(Lesson2)
ADD_SUBDIRECTORY(Lesson3)


######################################################################################


######################################################################################
### 4) Target to copy benchs and parameters file
######################################################################################

SET(BENCH_LIST
	020_05_01.txt
	020_05_02.txt
	020_10_01.txt
	020_10_02.txt
	020_20_01.txt
	050_05_01.txt
	050_10_01.txt
	050_20_01.txt
	100_05_01.txt
	100_10_01.txt
	100_20_01.txt
	200_10_01.txt		
)

ADD_CUSTOM_COMMAND(
    OUTPUT
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_02.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_02.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/200_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/README
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/020_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_01.txt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/020_05_02.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_02.txt    	
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/020_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_01.txt   	
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/020_10_02.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_02.txt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/020_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_20_01.txt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/050_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_05_01.txt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/050_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_10_01.txt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/050_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_20_01.txt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/100_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_05_01.txt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/100_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_10_01.txt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/100_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_20_01.txt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/200_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/200_10_01.txt
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/examples/flowshop/benchs/README
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/README
)

ADD_CUSTOM_COMMAND(
    OUTPUT ${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson1/Sch1.param
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson1/Sch1.param
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson1
)

ADD_CUSTOM_COMMAND(
    OUTPUT ${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson2/FlowShopEA.param
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson2/FlowShopEA.param
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson2
)

ADD_CUSTOM_COMMAND(
    OUTPUT ${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson3/FlowShopEA2.param
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson3/FlowShopEA2.param
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson3
)

ADD_CUSTOM_TARGET(install DEPENDS 
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson1/Sch1.param
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson2/FlowShopEA.param
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson3/FlowShopEA2.param
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_02.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_02.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_20_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_05_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_10_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_20_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_05_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_10_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_20_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/200_10_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/README
	
)
			  
ADD_CUSTOM_TARGET(benchs DEPENDS
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_02.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_02.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_20_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_05_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_10_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_20_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_05_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_10_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_20_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/200_10_01.txt
	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/README
)			  

ADD_CUSTOM_TARGET(lesson1 DEPENDS ${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson1/Sch1.param)

ADD_CUSTOM_TARGET(lesson2 DEPENDS ${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson2/FlowShopEA.param)

ADD_CUSTOM_TARGET(lesson3 DEPENDS ${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson3/FlowShopEA2.param)

