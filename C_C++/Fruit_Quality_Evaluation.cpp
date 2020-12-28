#include "Fruits_Segmentation.h"
#include "Fruits_Feature_Extraction.h"

using namespace cv;

//char InputFolder_Path[30];
//char OutputFolder_Path[30];

bool Cuda_Checked = false;

int main(void)
{
	int processing_method = 0;
	int Type_Device = 0;

	printf("[INFOR]: FRUITS SEGMENTATION: 1\n");
	printf("[INFOR]: FRUITS FEATURE EXTRACTION: 2\n");
	printf("[ENTER]: 1 or 2? ");
	scanf("%d", &processing_method);
	printf("[INFOR]: CPU: 1\n");
	printf("[INFOR]: GPU-CUDA: 2\n");
	printf("[ENTER]: 1 or 2? ");
	scanf("%d", &Type_Device);

	if (Type_Device == 2)
	{
		Cuda_Checked = true;
		printf("[INFOR]: Run on GPU-CUDA.\n");
	}
	else
	{
		Cuda_Checked = false;
		printf("[INFOR]: Run on CPU.\n");
	}

	switch (processing_method)
	{
	case 1:
		Fruits_Segmentation();
		break;
	case 2:
		Fruits_Feature_Extraction();
		break;
	}

	//waitKey(0);

	return 0;
}
