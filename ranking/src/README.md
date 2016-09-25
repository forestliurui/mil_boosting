All commands are assumed to be run from the root directory of this repository, i.e. the one containing 'ranking'

(1) To run the server:

python ranking/src/server_ranking.py results_root_dir ranker_name dataset_category

(2) To run the client:
python ranking/src/client_ranking.py server_URL/IP_address

or 

./ranking/src/start_client_noCV.sh [server_URL/IP_address]
(the shell script will start over each client program after every task)

(3) To get the statistics results stored in csv file:
 i)For LETOR and MovieLen 
python ranking/src/statistics_boosting_multiple_ranking.py method_name dataset_category outputfile_name

ii)For UCI
python ranking/src/statistics_boosting_multiple_ranking_UCI.py method_name outputfile_name

(4) To postprocess the statistics, like ploting, get rank scores for Critical Difference Diagram, use 
python ranking/src/postprocess_statistics_boosting_ranking directory outputfile_name func_invoked 

See each source file for detailed usage information.