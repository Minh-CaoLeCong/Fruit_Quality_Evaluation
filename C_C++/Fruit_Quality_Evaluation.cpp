#include "Fruits_Segmentation.h"
#include "Fruits_Feature_Extraction.h"

using namespace cv;

//char InputFolder_Path[30];
//char OutputFolder_Path[30];

int main(void)
{
	int processing_method = 0;

	printf("[INFOR]: FRUITS SEGMENTATION: 1\n");
	printf("[INFOR]: FRUITS FEATURE EXTRACTION: 2\n");
	printf("[ENTER]: 1 or 2? ");
	scanf("%d", &processing_method);

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
