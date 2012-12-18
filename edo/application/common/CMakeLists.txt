PROJECT(common)

SET(RESOURCES
  gplot.py
  ggobi.py
  boxplot_eda_n_edasa.py
  )

FOREACH(file ${RESOURCES})
  EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/${file}
    ${EDO_BIN_DIR}/${file}
    )
ENDFOREACH(file)
