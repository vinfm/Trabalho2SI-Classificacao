CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
ROOT := ..
TARGET := aprendizagem_conexionista

SRC := \
	main_rede.cpp \
	util.cpp \
	Layer.cpp \
	Neuron.cpp \
	NeuralNet.cpp \
	NeuralNetBuilder.cpp \
	$(ROOT)/Dataset.cpp \
	$(ROOT)/Metrics.cpp

OBJ := $(SRC:.cpp=.o)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(ROOT) -I. -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(ARGS)

clean:
	rm -f $(OBJ) $(TARGET)
