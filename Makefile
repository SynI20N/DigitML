:WARNINGS = -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wformat=2\
 -Winit-self -Wmissing-declarations -Wredundant-decls -Wshadow\
 -Wstrict-overflow=5 -Wswitch-default -Wundef

FLAGS = $(WARNINGS) -std=c++14

SRC = src/main.cpp

sigmoid: 
	g++ $(FLAGS) -Ofast $(SRC) -I include -o main

bent_identity:
	g++ $(FLAGS) -DPERS -Ofast $(SRC) -I include -o main

test:
	g++ $(FLAGS) -DTESTS -Ofast $(SRC) -I include -o main -lgtest

coverage:
	g++ $(FLAGS) -DTESTS -fprofile-arcs -ftest-coverage -O0 -g $(SRC) -I include -o main -lgtest

all: sigmoid

debug:
	g++ $(FLAGS) -DDEBUG $(SRC) -o main
	./main