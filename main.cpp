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
#include <chrono>

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

struct Bayes {
    valarray<valarray<int> > pixel;
    valarray<int> label;
    valarray<valarray<valarray<double> > > p;
    valarray<double> q;
    const int H = 4;

    Bayes(string image_file, string label_file) {
        input_images(image_file, pixel);
        input_labels(label_file, label);
        p.resize(10);
        for (int i = 0; i < 10; ++i) p[i].resize(pixel[0].size());
        for (int i = 0; i < 10; ++i) for (int j = 0; j < p[i].size(); ++j) p[i][j].resize(2);
        q.resize(10);
    }

    void init() {
        for (int i = 0; i < label.size(); ++i) q[label[i]] += 1;
        for (int i = 0; i < 10; ++i) q[i] /= label.size(), q[i] = log(q[i]);
        valarray<int> cnt(10);
        for (int i = 0; i < label.size(); ++i) {
            for (int j = 0; j < pixel[i].size(); ++j) {
                p[label[i]][j][(pixel[i][j] > H)] += 1;
                ++cnt[label[i]];
            }
        }
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < pixel[i].size(); ++j) {
                p[i][j][0] /= cnt[i];
                p[i][j][0] = log(p[i][j][0]);
                p[i][j][1] /= cnt[i];
                p[i][j][1] = log(p[i][j][1]);
            }
        }
    }

    int recognize(const valarray<int> &new_image) {
        int ans = 0;
        double mx = q[0];
        for (int j = 0; j < new_image.size(); ++j) mx += p[0][j][(new_image[j] > 0)];
        for (int i = 1; i < 10; ++i) {
            double now = q[i];
            for (int j = 0; j < new_image.size(); ++j) {
                now += p[i][j][(new_image[j] > H)];
            }
            if (now > mx) mx = now, ans = i;
        }
        return ans;
    }
};


//========================= solutions

void knn_solution(const valarray<valarray<int> > &test_pixel, const valarray<int> &test_label) {
    KNN knn(type[train] + image_file, type[train] + label_file);
    knn.init();
    int cnt = 0, i = 0;
    int N = test_pixel.size();
    vector<int> v(N);
    // for_each(std::execution::par_unseq, v.begin(), v.end(), [&]() {
    //     int re = knn.recognize(test_pixel[i]);
    //     if (re != test_label[i]) cout << "NO. " << i << " is recognized as: " << re << 
    //         " and actually is: " << test_label[i] << endl;
    //     cnt += (re == test_label[i]);
    //     ++i;
    // });
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int re = knn.recognize(test_pixel[i]);
#ifdef none
        cout << "NO. " << i << " is recognized as: " << re << 
            " and actually is: " << test_label[i] << endl;
        if (re != test_label[i]) cout << "NO. " << i << " is recognized as: " << re << 
            " and actually is: " << test_label[i] << endl;
#endif
        cnt += (re == test_label[i]);
    }
    cout << "The accuracy rating is: " << setiosflags(ios::fixed) << setprecision(2) << 1.0 * cnt / N << endl;
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
    cout << "The accuracy rating is: " << setiosflags(ios::fixed) << setprecision(2) << 1.0 * cnt / N << endl;
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
#ifdef none
    knn_solution(test_pixel, test_label);
#endif
#ifdef debug
    bayes_solution(test_pixel, test_label);
#endif
    return 0;
}