main: main.o read_VOL.o 
	g++ $(shell pkg-config --cflags --libs opencv) -o main read_VOL.o main.o

main.o: main.cpp read_VOL.h
	g++ -c -o main.o main.cpp

read_VOL.o: read_VOL.h read_VOL.cpp
	g++ -c -o read_VOL.o read_VOL.cpp
	
clean: 
	rm *.o main
