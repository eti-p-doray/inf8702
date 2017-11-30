/**
 This file is part of Poisson Image Editing.
 
 Copyright Christoph Heindl 2015
 
 Poisson Image Editing is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 Poisson Image Editing is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with Poisson Image Editing.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <blend/clone.h>
#include <opencv2/opencv.hpp>

#include <chrono>

/**
 
 Naive image cloning by just copying the values from foreground over background
 
 */
void naiveClone(cv::InputArray background_,
                cv::InputArray foreground_,
                cv::InputArray foregroundMask_,
                int offsetX, int offsetY,
                cv::OutputArray destination_)
{
    cv::Mat bg = background_.getMat();
    cv::Mat fg = foreground_.getMat();
    cv::Mat fgm = foregroundMask_.getMat();
    
    destination_.create(bg.size(), bg.type());
    cv::Mat dst = destination_.getMat();
    
    cv::Rect overlapAreaBg, overlapAreaFg;
    blend::detail::findOverlap(background_, foreground_, offsetX, offsetY, overlapAreaBg, overlapAreaFg);
    
    bg.copyTo(dst);
    fg(overlapAreaFg).copyTo(dst(overlapAreaBg), fgm(overlapAreaFg));
    
}

template <class F>
double benchmark(const F& fcn, int nb_run = 3) {
  double avg = 0;
  for (int i = 0; i < nb_run; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    fcn();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    avg += diff.count();
  }
  avg /= nb_run;
  return avg;
}

/**
 
 Main entry point.
 
 */
int main(int argc, char **argv)
{
    if (argc != 7) {
        std::cerr << argv[0] << " background foreground mask offsetx offsety" << std::endl;
        return -1;
    }
    
    cv::Mat background = cv::imread(argv[1]);
    cv::Mat foreground = cv::imread(argv[2]);
    cv::Mat mask = cv::imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
    int offsetx = atoi(argv[4]);
    int offsety = atoi(argv[5]);
    
    
    cv::Mat result;
    
    naiveClone(background, foreground, mask, offsetx, offsety, result);
    cv::imshow("Naive", result);
    cv::imwrite("naive.png", result);
    blend::CloneType method = static_cast<blend::CloneType>(atoi(argv[6]));
  
    std::cout << benchmark([&](){
      blend::seamlessClone(background, foreground, mask, offsetx, offsety, result, method);
    }) << std::endl;
    cv::imwrite("result-ref.jpg", result);
    
    return 0;
}




