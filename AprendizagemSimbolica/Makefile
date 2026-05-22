CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
ROOT := ..
TARGET := aprendizagem_simbolica
BLIND_TARGET := blind_test_simulado

SRC := \
	main.cpp \
	Cart.cpp \
	Criteria.cpp \
	DecisionTree.cpp \
	Forest.cpp \
	ForestBuilder.cpp \
	Node.cpp \
	Splitter.cpp \
	TreeBuilder.cpp \
	$(ROOT)/Metrics.cpp \
	$(ROOT)/Dataset.cpp

BLIND_SRC := \
	BlindTest.cpp \
	$(ROOT)/Dataset.cpp

OBJ := $(SRC:.cpp=.o)
BLIND_OBJ := $(BLIND_SRC:.cpp=.o)

.PHONY: all run run-blind-test clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BLIND_TARGET): $(BLIND_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(ROOT) -I. -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(ARGS)

run-blind-test: $(BLIND_TARGET)
	./$(BLIND_TARGET) $(ARGS)

clean:
	rm -f $(OBJ) $(BLIND_OBJ) $(TARGET) $(BLIND_TARGET)
