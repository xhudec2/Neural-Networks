SRC = src/*.cpp
FLAGS = -std=c++11 -Wall -Wextra -Werror -pedantic -g

all:
	g++ $(FLAGS) $(SRC) -o net
