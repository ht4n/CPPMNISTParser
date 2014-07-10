#include "MNISTParser.h"

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        printf("Syntax: <program.exe> <image-file> <label-file>");
        return 1;
    }

    MNISTDataset mnist;
    assert(0 == mnist.Parse(argv[1], argv[2]));
    mnist.Print();

    return 0;
}