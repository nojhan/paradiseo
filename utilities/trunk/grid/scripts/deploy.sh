#! /bin/bash

function exe()
{
    if  [ -e $1 ];
    then
	kadeploy -e paradiseo -f $1 -p $2
	rm $1
	PIDS="$PIDS $!"
    fi
}

`cat $OAR_NODEFILE | uniq > machines`
NODEFILE=machines
PIDS=""
N=0
for i in `cat machines`;
do
  name=${i%-*r}
  if [ "$name" = "azur" ];then
      echo $i >>machineAzur
      let $[N+=1]
  fi
  
  if [ "$name" = "sol" ];then
      echo $i >>machineSol
      let $[N+=1]
  fi

  if [ "$name" = "helios" ];then
      echo $i >>machineHelios
      let $[N+=1]
  fi

  if [ "$name" = "chti" ];then
      echo $i >>machineChti
      let $[N+=1]
  fi

  if [ "$name" = "chicon" ];then
      echo $i >>machineChicon
      let $[N+=1]
  fi

  if [ "$name" = "capricorne" ];then
      echo $i >>machineCapricorne
      let $[N+=1]
  fi

  if [ "$name" = "sagittaire" ];then
      echo $i >>machineSagittaire
      let $[N+=1]
  fi
  
  if [ "$name" = "gdx" ];then
      echo $i >>machineGdx
      let $[N+=1]
  fi

  if [ "$name" = "node" ];then
      echo $i >>machineNode
      let $[N+=1]
  fi

  if [ "$name" = "grelon" ];then
      echo $i >>machineGrelon
      let $[N+=1]
  fi

  if [ "$name" = "grillon" ];then
      echo $i >>machineGrillon
      let $[N+=1]
  fi

done

exe machineSol sda3
exe machineAzur hda3
exe machineHelios sda3
exe machineChti sda3
exe machineChicon sda3
exe machineCapricorne hda3
exe machineSagittaire sda3
exe machineGdx sda3
exe machineNode sda3
exe machineGrelon sda3
exe machineGrillon sda3
wait $PIDS
MASTER=`head -n 1 $OAR_NODEFILE`
scp -o StrictHostKeyChecking=no machines root@${MASTER}:/home/g5k/machines
echo "You can connect to the Master :  $MASTER"


