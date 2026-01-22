@echo off
setlocal

g++ -c -O3 -ffast-math -march=native -pthread -fopenmp -std=c++23 -mavx2 src/GF2N.cpp -Iinclude -o build/GF2N.o

g++ -c -I "C:\\dev\\lib\\eigen" -O3 -ffast-math -march=native -pthread -fopenmp -std=c++23 src/sym_space.cpp -Iinclude -o build/sym_space.o

g++ -c -I "C:\\dev\\lib\\eigen" -O3 -ffast-math -march=native -pthread -fopenmp -std=c++23 src/Qfunc.cpp -Iinclude -o build/Qfunc.o

g++ -c -I "C:\\dev\\lib\\eigen" -O3 -ffast-math -march=native -pthread -fopenmp -std=c++23 src/displaced_Qfunc.cpp -Iinclude -o build/displaced_Qfunc.o

g++ -c -I "C:\\dev\\lib\\eigen" -O3 -ffast-math -march=native -pthread -fopenmp -std=c++23 src/states.cpp -Iinclude -o build/states.o

g++ -c -I "C:\\dev\\lib\\eigen" -O3 -ffast-math -march=native -pthread -fopenmp -std=c++23 src/kravchuk.cpp -Iinclude -o build/kravchuk.o

ar rcs build/libdiscrete_space.a build/GF2N.o build/sym_space.o build/Qfunc.o build/displaced_Qfunc.o build/states.o

pause

endlocal