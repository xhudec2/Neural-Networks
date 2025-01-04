SRC = src/*.cpp src/matrix/*.cpp src/parser/*.cpp src/data/*.cpp src/tests/*.cpp src/network/*.cpp

FLAGS = -std=c++20 -g -Wall # -Wextra -Werror -pedantic
COMPILER = g++
ifeq ($(shell uname), Darwin)
	COMPILER=/opt/homebrew/opt/llvm/bin/clang++
endif

release: FLAGS +=  -O3 -flto -funroll-loops -march=native -fno-rtti -ffast-math -fprefetch-loop-arrays
# -flto -fprefetch-loop-arrays -fno-rtti -ffast-math

all:
	@$(COMPILER) $(FLAGS) $(SRC) -o net
	@echo "compilation done"

release: all

