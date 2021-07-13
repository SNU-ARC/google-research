CPPFLAGS=-Ofast -march=native -flto -std=c++14 -fopenmp -fpic -Wpedantic -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
INCLUDEFLAGS=-I src/lib/include -I src/lib/external/eigen -I src/lib/external/simple-serializer

all: datagen.cc
	g++ $(CPPFLAGS) -o  bin/datagen datagen.cc

clean: 
	rm -f bin/datagen
# {"mode":"full","isActive":false}