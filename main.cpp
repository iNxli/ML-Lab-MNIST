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
#include <bitset>

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
#include "svm.h"

//========================= solutions

void knn_solution(const valarray<valarray<int> > &test_pixel, const valarray<int> &test_label) {
    KNN knn(type[train] + image_file, type[train] + label_file);
    knn.init();
    int cnt = 0;
    int N = test_pixel.size();
    for (int i = 0; i < N; ++i) {
        int re = knn.recognize(test_pixel[i]);
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
        cnt += (re == test_label[i]);
    }
    cout << "The accuracy rating is: " << setiosflags(ios::fixed) << setprecision(6) << 1.0 * cnt / N << endl;
}

void svm_solution(const valarray<valarray<int> > &test_pixel, const valarray<int> &test_label) {
    SVM svm(type[train] + image_file, type[train] + label_file);
    svm.init();
    vector<int> cnt(10, 0);
    int N = 500, _cnt = 0;
    for (int i = 0; i < N; ++i) {
        cout << i << endl;
        cnt.clear();
        cnt.resize(10, 0);
        for (int j = 0; j < 10; ++j) {
            for (int k = j + 1; k < 10; ++k) {
                int re = svm.recognize(j, k, test_pixel[i]);
                ++cnt[re];
            }
        }
        int ans, mx = 0;
        for (int i = 0; i < 10; ++i) {
            if (cnt[i] > mx) mx = cnt[i], ans = i;
        }
        _cnt += (ans == test_label[i]);
    }
    cout << "The accuracy rating is: " << setiosflags(ios::fixed) << setprecision(2) << 1.0 * _cnt / N << endl;
}

//========================= main function

int main() {
    ios::sync_with_stdio(false);
    valarray<valarray<int> > test_pixel;
    valarray<int> test_label;
    // input api
    input_images(type[test] + image_file, test_pixel);
    input_labels(type[test] + label_file, test_label);

    auto start = clock();
    
    knn_solution(test_pixel, test_label);
    
    auto finish = clock();
    cout << "Running time: " << 1.0 * (finish - start) / CLOCKS_PER_SEC << "\n";

#ifdef none
    bayes_solution(test_pixel, test_label);
    svm_solution(test_pixel, test_label);
#endif
    return 0;
}