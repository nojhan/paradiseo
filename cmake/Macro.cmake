######################################################################################
### add_lesson(module target files)
### Macro to add a lesson to a specific module. 
### Currently module must be "mo" or "moeo".
### The target name will be prefixed by module name.
### Paramaters files must have the same name as cpp file. No need to have a .param
### file. CMake will check if such a file exists.
######################################################################################

macro(add_lesson module target files)
    foreach(i ${files})
        if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${i}.param)
            add_executable(${i} ${i}.cpp)
        else(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${i}.param)
            if(${CMAKE_VERBOSE_MAKEFILE})
                message(STATUS "Copying ${i}.param")
            endif(${CMAKE_VERBOSE_MAKEFILE})
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
			        ${CMAKE_CURRENT_SOURCE_DIR}/${i}.param
			        ${CMAKE_CURRENT_BINARY_DIR}/${i}.param)
	        add_executable(${i}
		        ${i}.cpp
		        ${CMAKE_CURRENT_BINARY_DIR}/${i}.param)
        endif(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${i}.param)
        if(${module} MATCHES mo)
	        target_link_libraries(${i} eoutils ga eo)
	    elseif(${module} MATCHES moeo)
	        target_link_libraries(${i} moeo flowshop eo eoutils)
	    elseif(${module} MATCHES smp)
	        target_link_libraries(${i} smp eo eoutils)
	    endif() 
	    install(TARGETS ${i} RUNTIME DESTINATION share/${PROJECT_TAG}/${module}/tutorial/${target} COMPONENT examples)   
	endforeach(i)
	
	# Custom target
	add_custom_target(${module}${target} DEPENDS
			${files}
			${files}.param)
endmacro()
