SRC = src/*.cpp

FLAGS = -std=c++20 -Wall -Wextra -Werror -pedantic -g
release: FLAGS += -O3 -funroll-loops -march=native 
# -flto -fprefetch-loop-arrays -fno-rtti -ffast-math

all:
	g++ $(FLAGS) $(SRC) -o net

release: all

