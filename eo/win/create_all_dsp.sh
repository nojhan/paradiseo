#! /bin/tcsh -f
# 
# script that generates the DSP files for all programs in test dir
#
# to add a new one, add the corresponding line:
#
# create_dsp SourceFileName dspFileName [optional additional library]
#
# don't use upper-case letters in the dspFileName

# create a backup
/bin/mv eo.dsw eo.dsw~
# restore empty DSW
cp eo.org eo.dsw
# GO for all programs
create_dsp.sh t-eoParetoFitness  t_eoparetofitness 
create_dsp.sh t-eoPareto         t_eopareto 
create_dsp.sh t-eofitness        t_eofitness 
create_dsp.sh t-eoRandom         t_eorandom 
create_dsp.sh t-eobin            t_eobin 
create_dsp.sh t-eoVirus          t_eovirus 
create_dsp.sh t-MGE              t_mge 
create_dsp.sh t-MGE1bit          t_mge1bit 
create_dsp.sh t-MGE-control      t_mge-control 
create_dsp.sh t-eoStateAndParser t_eostateandparser 
create_dsp.sh t-eoCheckpointing  t_eocheckpointing 
create_dsp.sh t-eoSSGA           t_eossga 
create_dsp.sh t-eoExternalEO     t_eoexternaleo 
create_dsp.sh t-eoSymreg         t_eosymreg 
create_dsp.sh t-eo               t_eo 
create_dsp.sh t-eoReplacement    t_eoreplacement 
create_dsp.sh t-eoSelect         t_eoselect 
create_dsp.sh t-eoGenOp          t_eogenop 
create_dsp.sh t-eoGA             t_eoga             ga
create_dsp.sh t-eoReal           t_eoreal           es
create_dsp.sh t-eoVector         t_eovector 
create_dsp.sh t-eoESAll          t_eoesall          es
create_dsp.sh t-eoPBIL           t_eopbil           ga
