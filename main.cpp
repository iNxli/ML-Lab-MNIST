#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <cmath>
#include <iomanip>
#include <random>
#include <ctime>
#include <set>
#include <valarray>
#include <omp.h>

using namespace std;

#define debug

//========================= file input

#include "input_api.h"

//========================= data initialise

enum {train, test} option;
string type[2] = {"train", "t10k"};
string image_file = "-images.idx3-ubyte";
string label_file = "-labels.idx1-ubyte";

//========================= algorithms

#include "knn.h"
#include "bayes.h"

//========================= solutions

void knn_solution(const valarray<valarray<int> > &test_pixel, const valarray<int> &test_label) {
    KNN knn(type[train] + image_file, type[train] + label_file);
    knn.init();
    int cnt = 0;
    int N = test_pixel.size();
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int re = knn.recognize(test_pixel[i]);
#ifdef none
        cout << "NO. " << i << " is recognized as: " << re << 
            " and actually is: " << test_label[i] << endl;
#endif
        if (re != test_label[i]) cout << "NO. " << i << " is recognized as: " << re << 
            " and actually is: " << test_label[i] << endl;
        cnt += (re == test_label[i]);
    }
    cout << "The accuracy rating is: " << setiosflags(ios::fixed) << setprecision(6) << 1.0 * cnt / N << endl;
}

void bayes_solution(const valarray<valarray<int> > &test_pixel, const valarray<int> &test_label) {
    Bayes bayes(type[train] + image_file, type[train] + label_file);
    bayes.init();
    int cnt = 0;
    int N = test_pixel.size();
    for (int i = 0; i < N; ++i) {
        int re = bayes.recognize(test_pixel[i]);
#ifdef none
        cout << "NO. " << i << " is recognized as: " << re << 
            " and actually is: " << test_label[i] << endl;
#endif
        if (re != test_label[i]) cout << "NO. " << i << " is recognized as: " << re << 
            " and actually is: " << test_label[i] << endl;
        cnt += (re == test_label[i]);
    }
    cout << "The accuracy rating is: " << setiosflags(ios::fixed) << setprecision(6) << 1.0 * cnt / N << endl;
}

//========================= main function

int main() {
    valarray<valarray<int> > test_pixel;
    valarray<int> test_label;
    input_images(type[test] + image_file, test_pixel);
    input_labels(type[test] + label_file, test_label);
#ifdef none
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            cout << (test_pixel[195][i * 28 + j] > 0);
        }
        cout << endl;
    }
#endif
#ifdef debug
    knn_solution(test_pixel, test_label);
#endif
#ifdef none
    bayes_solution(test_pixel, test_label);
#endif
    return 0;
}