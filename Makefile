#Makefile Eye Detection Kernel Descriptor
EDKD_LIB=lib
EDKD_UTIL=utils
EDKD_SRC=src
CFLAGS= -Wall -ggdb -I include -I lib -I /usr/local/include
LDFLAGS= /usr/local/lib/libopencv_objdetect.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_ml.so
GCC=g++
EDKD_MAKE=make
EDKD_TARGET=edkd
EDKD_BIN_APP=/home/diego/Documentos/Codigos/Proyectos-C/source_code_maestria/bin/

#OBJS	= ${SRCS:.c=.o}

.SUFFIXES:
.SUFFIXES: .o .cpp

.c.o:
	$(GCC) $(CFLAGS) -o $@ -c $< 

all:
	$(EDKD_MAKE) -C $(EDKD_LIB)
	$(GCC) $(CFLAGS) $(EDKD_LIB)/*.o  src/main.cpp -o $(EDKD_TARGET) $(LDFLAGS)
	cp $(EDKD_TARGET) $(EDKD_BIN_APP)

run:
	#$(EDKD_BIN_APP)  -l /root/workspace/altparserV11/app/var/log/altparserV11.log  -p /root/workspace/altparserV11/app/var/run/altparserV11.pid -c /root/workspace/altparserV11/app/etc/altparserV11.conf -b /root/workspace/altparserV11/app/var/lck/altparserV11.lck -n altparserV11 -i
	
trace:
	#strace $(ALTPARSERV11_BIN_APP)  -l /root/workspace/altparserV11/app/var/log/altparserV11.log  -p /root/workspace/altparserV11/app/var/run/altparserV11.pid -c /root/workspace/altparserV11/app/etc/altparserV11.conf -b /root/workspace/altparserV11/app/var/lck/altparserV11.lck -n altparserV11 -i
	
gdb:
	gdb $(EDKD_BIN_APP)

clean:
	$(EDKD_MAKE) clean -C $(EDKD_LIB)
	$(EDKD_MAKE) clean -C $(EDKD_SRC)
	rm -f $(EDKD_TARGET)
	rm -f $(EDKD_BIN_APP)
	#rm -f /root/workspace/altparserV11/app/var/run/altparserV11.pid
	#rm -f /root/workspace/altparserV11/app/var/lck/altparserV11.lck
