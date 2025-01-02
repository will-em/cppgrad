CXX = g++
CXXFLAGS = -I./src -Wall -Wextra -std=c++17
SRC = src/main.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = cppgrad

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
