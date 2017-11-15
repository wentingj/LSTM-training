#!/bin/bash
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared lstm_trainingOP.cc -o lstm_trainingOP.so -fPIC -I $TF_INC -O2 -I/home/wentingj/software/anaconda3/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -L$TF_LIB -ltensorflow_framework
#g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -O2 -L$TF_LIB -ltensorflow_framework
#g++ -std=c++11 -shared lstm.cc -o lstm.so -fPIC -I $TF_INC -O2 -L$TF_LIB -ltensorflow_framework

#python tf_rnn_benchmarks_cpu_lstm_backward_check.py -n basic_lstm -i 3 -l 2 -s 2 -b 64
python tf_rnn_benchmarks_cpu_lstm_backward_check.py -n basic_lstm -i 150 -l 1024 -s 10 -b 64
#python test_lstm.py
