echo Running ensemble of $1 simulations
EXACA_EXEC="./build/install/bin/ExaCA"
EXACA_ANALYSIS_EXEC="./build/install/bin/ExaCA-GrainAnalysis"

END=$1
for ((i=0;i<END;i++));
do
    mpiexec -n 4 ${EXACA_EXEC} examples/Inp_Case$i.json
    ${EXACA_ANALYSIS_EXEC} analysis/examples/AnalyzeDirSCalibration.json Case$i
done
wait
