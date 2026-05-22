CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

.PHONY: all redes arvore run-redes run-arvore clean clean-redes clean-arvore

all: redes arvore

redes:
	$(MAKE) -C AprendizagemConexionista

arvore:
	$(MAKE) -C AprendizagemSimbolica

run-redes:
	$(MAKE) -C AprendizagemConexionista run ARGS="$(ARGS)"

run-arvore:
	$(MAKE) -C AprendizagemSimbolica run ARGS="$(ARGS)"

clean-redes:
	$(MAKE) -C AprendizagemConexionista clean

clean-arvore:
	$(MAKE) -C AprendizagemSimbolica clean

clean: clean-redes clean-arvore
