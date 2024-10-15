SRC = src/*.cpp
FLAGS = -std=c++20 -Wall -Wextra -Werror -pedantic -g

all:
	g++ $(FLAGS) $(SRC) -o net
