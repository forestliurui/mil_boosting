This is for how to perform experiments on MIL or supervised learning classification problems.
All commands are assumed to be run from the root directory of this repository, i.e. the one containing the directory of 'ranking' .

(1) To run the server program:
 i)The distributed version 
 ( For each boosting round, server program is responsible for cross validation to select parameters for best classifier using the inner layer folds )
python src/server_boosting.py configuration_file results_root_diretory

ii)The nondistributed version 
( No Cross Validation is performed while running the experiments. Global variable INNER_CROSS_VALIDATION is set False to skip inner layer folds when creating tasks )
python src/server_boosting_noCV.py configuration_file results_root_diretory

(2) To run the client program:
 i)The distributed version 
python src/client.py server_URL/IP_address
ii)The nondistributed version  
python src/client_noCV.py server_URL/IP_address

or 

./src/start_client_noCV.sh [server_URL/IP_address]
(the shell script will start over each client program after every task)

(3) To get the statistics results stored in csv file:
python src/statistics_boosting_multiple.py configuration_file results_root_directory outputfile_raw_name

(4) To postprocess the statistics, like ploting, get rank scores for Critical Difference Diagram, use 
python src/postprocess_statistics_boosting.py directory outputfile_name func_invoked 

Note:
For experiments on synthetic datasets, use code in src/synthetic_experiment_boosting.py

See each source file for detailed usage information.