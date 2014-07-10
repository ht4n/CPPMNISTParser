/*
    Copyright 2014 Henry Tan
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
    http ://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory>
#include <assert.h>

//
// C++ MNIST dataset parser
// 
// Specification can be found in http://yann.lecun.com/exdb/mnist/
//
class MNISTDataset final
{
public:
    MNISTDataset()
        : m_count(0),
        m_width(0),
        m_height(0),
        m_imageSize(0),
        m_buffer(nullptr),
        m_imageBuffer(nullptr),
        m_categoryBuffer(nullptr)
    {
    }

    ~MNISTDataset()
    {
        if (m_buffer) free(m_buffer);
    }

    void Print()
    {
        for (size_t n = 0; n < m_count; ++n)
        {
            const float* imageBuffer = &m_imageBuffer[n * m_imageSize];
            for (size_t j = 0; j < m_height; ++j)
            {
                for (size_t i = 0; i < m_width; ++i)
                {
                    printf("%3d ", (uint8_t)imageBuffer[j * m_width + i]);
                }
                printf("\n");
            }

            printf("\n ===> cat(%d)\n\n", n, m_categoryBuffer[n]);
        }
    }

    size_t GetImageWidth() const
    {
        return m_width;
    }

    size_t GetImageHeight() const
    {
        return m_height;
    }

    size_t GetImageCount() const
    {
        return m_count;
    }

    size_t GetImageSize() const
    {
        return m_imageSize;
    }

    const float* GetImageData() const
    {
        return m_imageBuffer;
    }

    const float* GetCategoryData() const
    {
        return m_categoryBuffer;
    }

    //
    // Parse MNIST dataset
    // Specification of the dataset can be found in:
    // http://yann.lecun.com/exdb/mnist/
    //
    int Parse(const char* imageFile, const char* labelFile)
    {
        FILE* fimg = nullptr;
        if (0 != fopen_s(&fimg, imageFile, "rb"))
        {
            printf("Failed to open %s for reading\n", imageFile);
            return 1;
        }
        std::shared_ptr<FILE> autofimg(fimg, [](FILE* f) { if (f) { fclose(f); }});

        FILE* flabel = nullptr;
        if (0 != fopen_s(&flabel, labelFile, "rb"))
        {
            printf("Failed to open %s for reading\n", labelFile);
            return 1;
        }
        std::shared_ptr<FILE> autoflabel(flabel, [](FILE* f) { if (f) { fclose(f); }});

        uint32_t value;

        // Read magic number
        assert(!feof(fimg));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, fimg);
        printf("Image Magic        :%0X(%I32u)\n", _byteswap_ulong(value), _byteswap_ulong(value));
        assert(_byteswap_ulong(value) == 0x00000803);

        // Read count
        assert(!feof(fimg));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, fimg);
        const uint32_t count = _byteswap_ulong(value);
        printf("Image Count        :%0X(%I32u)\n", count, count);
        assert(count > 0);

        // Read rows
        assert(!feof(fimg));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, fimg);
        const uint32_t rows = _byteswap_ulong(value);
        printf("Image Rows         :%0X(%I32u)\n", rows, rows);
        assert(rows > 0);

        // Read cols
        assert(!feof(fimg));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, fimg);
        const uint32_t cols = _byteswap_ulong(value);
        printf("Image Columns      :%0X(%I32u)\n", cols, cols);
        assert(cols > 0);

        // Read magic number (label)
        assert(!feof(flabel));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, flabel);
        printf("Label Magic        :%0X(%I32u)\n", _byteswap_ulong(value), _byteswap_ulong(value));
        assert(_byteswap_ulong(value) == 0x00000801);

        // Read label count
        assert(!feof(flabel));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, flabel);
        printf("Label Count        :%0X(%I32u)\n", _byteswap_ulong(value), _byteswap_ulong(value));
        // The count of the labels needs to match the count of the image data
        assert(_byteswap_ulong(value) == count);

        Initialize(cols, rows, count);

        size_t counter = 0;
        while (!feof(fimg) && !feof(flabel))
        {
            float* imageBuffer = &m_imageBuffer[counter * m_imageSize];

            for (size_t j = 0; j < m_height; ++j)
            {
                for (size_t i = 0; i < m_width; ++i)
                {
                    uint8_t pixel;
                    fread_s(&pixel, sizeof(uint8_t), sizeof(uint8_t), 1, fimg);

                    imageBuffer[j * m_width + i] = pixel;
                }
            }

            uint8_t cat;
            fread_s(&cat, sizeof(uint8_t), sizeof(uint8_t), 1, flabel);
            assert(cat >= 0 && cat <= 9);
            m_categoryBuffer[counter] = cat;

            ++counter;
        }

        return 0;
    }
private:
    void Initialize(const size_t width, const size_t height, const size_t count)
    {
        m_width = width;
        m_height = height;
        m_imageSize = m_width * m_height;
        m_count = count;

        const size_t bufferSize =
            m_count * m_width * m_height
            + m_count * c_categoryCount
            ;

        m_buffer = (float*)malloc(bufferSize * sizeof(float));
        m_imageBuffer = m_buffer;
        m_categoryBuffer = m_imageBuffer + (m_count * m_width * m_height);
    }

    // The total number of images
    size_t m_count;

    // Dimension of the image data
    size_t m_width;
    size_t m_height;
    size_t m_imageSize;

    float* m_imageBuffer;

    static const int c_categoryCount = 10;

    // 1-of-N label of the image data (N = 10) 
    float* m_categoryBuffer;

    // The entire buffers that stores both the image data and the category data
    float* m_buffer;
};
