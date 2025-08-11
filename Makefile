.PHONY: all clean

SHELL:=/bin/bash

all: gmres

gmres: ./build/bin/gmres-host

./build/bin/gmres-host:
	source ./scripts/env.sh
	cmake -B ./build -S ./ -DCMAKE_BUILD_TYPE=Release -DUSPARSE_UNIT_TEST_ENABLE=OFF -DUSPARSE_BENCHMARK_ENABLE=OFF -DGMRES_TOOLS_ENABLE=OFF
	cmake --build ./build -j 4
	ln -s -f ./build/bin/gmres-host ./gmres

clean:
	rm -rf ./build/ ./gmres
