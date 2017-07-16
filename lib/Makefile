CFLAGS= -Wall -ggdb
GCC=g++

SRCS=helpers.cpp       \
        tools-deteccion.cpp \
        stringtools.cpp 

OBJS=${SRCS:.cpp=.o}

all: ${OBJS}

.SUFFIXES:
.SUFFIXES: .o .cpp

.c.o:   $(GCC) $(CFLAGS) -o $@ -c $<

clean: rm -f $(OBJS)
