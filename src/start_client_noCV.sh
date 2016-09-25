#!/bin/bash
#This is the bash script to automatically run the one-time client_noCV.py, i.e. client_noCV_1time.py 
#if no hostname or IP address is provided as argument from command-line, the local hostname will be used 


if [ "$1" == "" ]; then
	ip_address_local=$(hostname -I | awk '{print $1}')
	server_name=${ip_address_local}
else
	server_name=$1
fi
#echo ${ip_address_local}

	
for ((i=1;i<=20;i=i+1))
do
	echo "Bash starts python script: ${i} for ${server_name}"
	python src/client_noCV_1time.py ${server_name}
        #loop_count=${loop_count}-1
done