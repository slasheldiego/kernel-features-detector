#include "stringtools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

char* finishedString(char* s){
   char* p = s + strlen(s);
   while ( p > s && isspace((unsigned char)(*--p)))
        *p = '\0';
   return s;
}

char* backFirstNoWhite(const char* s){
   while (*s && isspace((unsigned char)(*s)))
        s++;
   return (char*)s;
}

char** split_me(char* s, const char delem){
   char** result = 0;
   int count = 0;
   char* tmp = s;
   char* last_comma = 0;
   char delim[2];
   delim[0] = delem;
   delim[1] = 0;

   while(*tmp){
	if (delem == *tmp){
	   count++;
	   last_comma = tmp;
	}
	tmp++;
   }

   count += last_comma < (s + strlen(s) - 1);
   
   count++;

   result = (char**) malloc(sizeof(char*) * count);

   if (result){
	int idx = 0;
        char* token = strtok(s,delim);

	while (token){
	   assert(idx < count);
	   *(result + idx++) = strdup(token);
	   token = strtok(0,delim);
	}
	assert(idx == count - 1);
	*(result + idx) = 0;
   }

   return result;
}
