SRC = src/*.cpp src/matrix/*.cpp src/parser/*.cpp src/data/*.cpp src/tests/*.cpp src/network/*.cpp
FLAGS = -std=c++20 # -Wall -Wextra -Werror -pedantic -g
COMPILER = g++
ifeq ($(shell uname), Darwin)
	COMPILER=/opt/homebrew/opt/llvm/bin/clang++
endif

release: FLAGS += -O3 -funroll-loops -march=native 
# -flto -fprefetch-loop-arrays -fno-rtti -ffast-math

all:
	@$(COMPILER) $(FLAGS) $(SRC) -o net
	@echo "compilation done"

release: all

