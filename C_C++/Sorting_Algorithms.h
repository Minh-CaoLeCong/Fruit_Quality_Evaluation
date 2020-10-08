#ifndef SORTING_ALGORITHMS_H
#define SORTING_ALGORITHMS_H

#include <filesystem>
#include <Windows.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <typeinfo>
#include <direct.h>

extern void swap(int *xp, int *yp);
extern void bubbleSortMaxMin(int arr[], int n);
extern void bubbleSortMinMax(int arr[], int n);
extern void printArray(int arr[], int size);
extern int maxElementIndex(int arr[], int n);

using namespace std;

#endif // SORTING_ALGORITHMS_H