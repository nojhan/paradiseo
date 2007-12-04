if [ $# -eq 2 ];then
    oarsub -I -t deploy -l nodes=$1,walltime=0$2:00:00
else
    echo "Usage : launch.sh <node-number> <time>"
    exit 0;
fi

