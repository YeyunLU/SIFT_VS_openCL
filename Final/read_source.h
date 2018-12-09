#pragma once

#include <stdio.h> 
#include <stdlib.h> 
//#include <string.h> 
//#include <string> 
//#include <fstream> 
#include <malloc.h>

// utility function which reads a text source-code file and returns entire file as a string.
// note: memory is allocated by this function but must be released by caller.

char* read_source(/* input */ const char *file_name, /* output */ size_t* file_size)
{
	FILE *file;
	file = fopen(file_name, "rb");
	if (!file) {
		printf("Error: Failed to open file '%s'\n", file_name);
		return NULL;
	}

	if (fseek(file, 0, SEEK_END))
	{
		printf("Error: Failed to seek file '%s'\n", file_name);
		fclose(file);
		return NULL;
	}
	size_t size = ftell(file);
	if (size == 0)
	{
		printf("Error: Failed to check position on file '%s'\n", file_name);
		fclose(file);
		return NULL;
	}

	rewind(file);

	char *src = (char *)malloc(sizeof(char) * size + 1);
	if (!src)
	{
		printf("Error: Failed to allocate memory for file '%s'\n", file_name);
		fclose(file);
		return NULL;
	}
	printf("Reading file '%s' (size %ld bytes)\n", file_name, size);

	size_t res = fread(src, 1, sizeof(char) * size, file);
	if (res != sizeof(char) * size)
	{
		printf("Error: Failed to read file '%s'\n", file_name);
		fclose(file);
		free(src);
		return NULL;
	}

	src[size] = '\0'; // NULL terminated  
	fclose(file);

	*file_size = size;
	return src;
};
