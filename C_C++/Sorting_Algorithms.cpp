#include "Sorting_Algorithms.h"

void swap(int *xp, int *yp)
{
	int temp = *xp;
	*xp = *yp;
	*yp = temp;
}

// A function to implement bubble sort  
void bubbleSortMaxMin(int arr[], int n)
{
	int i, j;
	for (i = 0; i < n - 1; i++)

		// Last i elements are already in place  
		for (j = 0; j < n - i - 1; j++)
			if (arr[j] < arr[j + 1])
				swap(&arr[j], &arr[j + 1]);
}

void bubbleSortMinMax(int arr[], int n)
{
	int i, j;
	for (i = 0; i < n - 1; i++)

		// Last i elements are already in place  
		for (j = 0; j < n - i - 1; j++)
			if (arr[j] > arr[j + 1])
				swap(&arr[j], &arr[j + 1]);
}

/* Function to print an array */
void printArray(int arr[], int size)
{
	int i;
	for (i = 0; i < size; i++)
		cout << arr[i] << " ";
	cout << endl;
}

int maxElementIndex(int arr[], int n)
{
	int i;

	// Initialize maximum element 
	int maxElement = arr[0];
	int maxIndex = 0;

	// Traverse array elements from second and compare every element with current max  
	for (i = 1; i < n; i++)
		if (arr[i] > maxElement)
		{
			maxElement = arr[i];
			maxIndex = i;
		}
	return maxElement, maxIndex;
}