#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm> // std::replace 在这里
#include "adcensus_stereo.h"

using namespace std;

// 保存视差图
void write_disparity(std::string filename, float* disps, int _height, int _width, int min_disparity, int max_disparity)
{
    uint8_t* vis_data = new uint8_t[size_t(_height) * _width];
    float min_disp = float(_width), max_disp = -float(_width);
    for (int i = 0; i < _height; i++) {
        for (int j = 0; j < _width; j++) {
            const float disp = disps[i * _width + j];
            if (disp != INFINITY) {
                min_disp = std::fmin(min_disp, disp);
                max_disp = std::fmax(max_disp, disp);
            }
        }
    }
    std::cout << "write out disparity, min disp is:" << min_disp << " max disp is" << max_disp << std::endl;
    for (int i = 0; i < _height; i++) {
        for (int j = 0; j < _width; j++) {
            const float disp = disps[i * _width + j];
            if (disp == INFINITY) {
                vis_data[i * _width + j] = 0;
            }
            else {
                vis_data[i * _width + j] = static_cast<uint8_t>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }
    stbi_write_png(filename.c_str(), _width, _height, 1, (void*)vis_data, 0);
    delete[] vis_data;
}

int main()
{
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL * 2048); // 2GB
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda set heap size failed!");
        return 1;
    }

    // ✅ 替换为你自己的图片路径
    std::string left_img_path_str = "/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/dataset/left/0000001.jpg";
    std::string right_img_path_str = "/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/dataset/right/0000001.jpg";

    // 如果你路径有 \，就统一替换成 /
    std::replace(left_img_path_str.begin(), left_img_path_str.end(), '\\', '/');
    std::replace(right_img_path_str.begin(), right_img_path_str.end(), '\\', '/');

    int l_width, l_height, l_bpp;
    int r_width, r_height, r_bpp;

    uint8_t* image_left = stbi_load(left_img_path_str.c_str(), &l_width, &l_height, &l_bpp, 3);
    std::cout << "Left image width: " << l_width << " height: " << l_height << " channels: " << l_bpp << std::endl;

    uint8_t* image_right = stbi_load(right_img_path_str.c_str(), &r_width, &r_height, &r_bpp, 3);
    std::cout << "Right image width: " << r_width << " height: " << r_height << " channels: " << r_bpp << std::endl;

    if (!image_left || !image_right) {
        std::cerr << "Failed to load images!" << std::endl;
        return 1;
    }

    ADCensus_Option option;
    option.cross_L1 *= 2;
    option.cross_L2 *= 2;
    option.cross_t1 *= 2;
    option.cross_t2 *= 2;
    option.min_disparity = 0;
    option.max_disparity = 256; // 你可以根据自己的需要调整
    option.do_filling = false;
    option.height = l_height;
    option.width = l_width;
    option.lrcheck_thres = 5.0f;

    ADCensusStereo* stereo = new ADCensusStereo(option);
    stereo->SetOption(option);
    stereo->Init();
    stereo->SetComputeImg(image_left, image_right);
    stereo->Compute();

    float* image_disp_left = stereo->RetrieveLeftDisparity();
    write_disparity("output_disp_left.png", image_disp_left, option.height, option.width, option.min_disparity, option.max_disparity);

    delete stereo;
    cudaDeviceSynchronize();
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    stbi_image_free(image_left);
    stbi_image_free(image_right);

    return 0;
}
