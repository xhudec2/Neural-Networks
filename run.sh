#!/bin/bash
## change this file to your needs

# echo "Adding some modules"

# module add gcc-10.2

echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
# g++ -Wall -std=c++17 -O3 src/main.cpp src/file2.cpp -o network
# g++ -std=c++20 -Wall -O3 -flto -funroll-loops -march=native -fno-rtti -ffast-math -fprefetch-loop-arrays src/*.cpp src/matrix/*.cpp src/parser/*.cpp src/data/*.cpp src/tests/*.cpp src/network/*.cpp -o net

make release

echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
nice -n 19 ./net

# echo "#################"
# echo "   EVALUATING    "
# echo "#################"
echo "Evaluating the whole training dataset including validation data..."
python3 ./evaluator/evaluate.py ./train_predictions.csv ./data/fashion_mnist_train_labels.csv
echo "Evaluating test dataset..."
python3 ./evaluator/evaluate.py ./test_predictions.csv ./data/fashion_mnist_test_labels.csv