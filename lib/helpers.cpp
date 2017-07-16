#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <libconfig.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <stdarg.h>
#include <math.h>
#include <dirent.h>

#include "helpers.hpp"

void hora2(int i, char* hora){

        time_t nowhours;
        struct timeval tp;
        struct tm *timeinfo;

        gettimeofday(&tp, 0);
        nowhours = tp.tv_sec;
        timeinfo = localtime(&nowhours);

        if(      i == 0 ){
                sprintf(hora,"%i-%02i-%02i %02i:%02i:%02i.%03li%c", timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, tp.tv_usec/1000,'\0');
        }else if(i == 1){
                sprintf(hora,"%i%02i%02i_%02i%02i%02i.%03li%c", timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, tp.tv_usec/1000,'\0');
        }else if(i == 2){
                sprintf(hora,"%i%02i%02i%02i%02i%02i.%03li%c", timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, tp.tv_usec/1000,'\0');
        }

}

int getSizePaths( char* path_src ){
   DIR *dir;
   struct dirent *ent;
   int size = 0;
   if ((dir = opendir (path_src)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
           if( strcmp(ent->d_name,".") != 0  && strcmp(ent->d_name,"..") != 0 ){
              size++;
           }
        }
   }
   return size;
}

void getPathImages( char* path_src, char** paths, int size ){
   DIR *dir;
   struct dirent *ent;
   int i = 0;
   if ((dir = opendir (path_src)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
           if( strcmp(ent->d_name,".") != 0  && strcmp(ent->d_name,"..") != 0 ){
              *(paths + i) = (char*) malloc( size );
              strcpy( *(paths+i), ent->d_name );
              i++;
           }

        }

        *(paths+i) = 0;
   }
}

