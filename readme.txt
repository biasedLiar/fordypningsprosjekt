This is the repository for the source code for Elias Ekern Baird's Master Thesis.

Tests are run by running the main function from the files
genericClients/Mass_testing_client.py
and
genericExpandedClients/Mass_expanded_client

To run a file from script (such as when running with tmux), the files
script.sh
and
script2.sh
will run the Mass_testing_client.py and Mass_expanded_client respectively.

In order to run a test, the following variable must be set:
sys_path: the location the program is run from
LINUX: whether the program should format files using the linux file format.

To choose the settings of the program being run, set the following settings:
SEED_COUNT: the number of seeds to test, set to 100 in this report
MULTITHREADING: Whether to run the seeds in parallell.
EXPLORATION_RATES: the exploration thresholds tested
GAUSSIANS: The gaussian width tested

K_VALUES: The numbers of clusters used when running K_Means
SEARCH_TREE_DEPTH: The number of steps forward in time VST-EWQM predicts
SEGMENTS: The g-value used for state space transformation
EXPANDER_GAUSSIAN: Used to change the gaussian width of state space transformation
USE_SIGMOID_WEIGHTING: Decides whether weighted K-Means is run with sigmoid weighting or with linear weighting

The following boolean variables are used to set what versions of the program to be run.
RUN_BASIC_NO_LEARN   Runs the basic EWQM algorithm
RUN_KMEANS_UNWEIGHTED    Runs the K-Means EWQM algorithm
RUN_KMEANS_WEIGHTED   Runs the weighted EWQM algorithm
RUN_SEARCH_TREE     Runs the VST-EWQM algorithm
RUN_SEARCH_TREE_KMEANS    Runs the K-Means EWQM algorithm

Whether or not the state spaced is transformed depends upon if the program is being run from Mass_testing_client or Mass_expanded_client.




