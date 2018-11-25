CXX = g++
CXXFLAGS = -O2 -g
LDFLAGS = -ljack -lfftw3f

# CXX = clang++
# CXXFLAGS += -fsanitize=address
# LDFLAGS += -fsanitize=address

all: measure

clean:
	rm -f measure *.o

measure: measure.o
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<
