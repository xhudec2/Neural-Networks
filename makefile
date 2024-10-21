SRC = src/*.cpp src/matrix/*.cpp
FLAGS = -std=c++20 -Wall -Wextra -Werror -pedantic -g
COMPILER = g++
ifeq ($(shell uname), Darwin)
	COMPILER=/opt/homebrew/opt/llvm/bin/clang++
endif

release: FLAGS += -O3 -funroll-loops -march=native 
# -flto -fprefetch-loop-arrays -fno-rtti -ffast-math

all:
	$(COMPILER) $(FLAGS) $(SRC) -o net

release: all

