#define pii pair<int, int>
struct KNN {
    valarray<valarray<int> > pixel, compressed_pixel;
    valarray<int> label;
    const int K = 5;
    const int N = 20;
    const int MAX_N = 10000;
    const int IMAGE_N = 28;

    KNN(string image_file, string label_file) {
        input_images(image_file, pixel);
        input_labels(label_file, label);
        compressed_pixel.resize(MAX_N);
    }

    void compress(const valarray<int> &in_image, valarray<int> &out_image) {
        int window_size = IMAGE_N - N + 1;
        out_image.resize(N * N);
        valarray<int> pre_sum(IMAGE_N * IMAGE_N);
        for (int i = 0; i < IMAGE_N; ++i) {
            for (int j = 0; j < IMAGE_N; ++j) {
                pre_sum[i * IMAGE_N + j] = in_image[i * IMAGE_N + j];
                int m2 = j ? pre_sum[i * IMAGE_N + (j - 1)] : 0;
                int m3 = i ? pre_sum[(i - 1) * IMAGE_N + j] : 0;
                int m1 = (i && j) ? pre_sum[(i - 1) * IMAGE_N + (j - 1)] : 0;
                pre_sum[i * IMAGE_N + j] += m2 + m3 - m1;
            }
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int m4 = pre_sum[(i + window_size - 1) * IMAGE_N + (j + window_size - 1)];
                int m2 = j ? pre_sum[(i + window_size - 1) * IMAGE_N + j - 1] : 0;
                int m3 = i ? pre_sum[(i - 1) * IMAGE_N + (j + window_size - 1)] : 0;
                int m1 = (i && j) ? pre_sum[(i - 1) * IMAGE_N + j - 1] : 0;
                out_image[i * N + j] = m4 - m2 - m3 + m1;
            }
        }
    }

    void init() {
        for (int i = 0; i < MAX_N; ++i) compress(pixel[i], compressed_pixel[i]);
    }

    int recognize(const valarray<int> &new_image) {
        set<pii> q; 
        valarray<int> new_com_image;
        compress(new_image, new_com_image);
#pragma omp parallel for
        for (int i = 0; i < MAX_N; ++i) {
            int dist = 0;
            for (int j = 0; j < N * N; ++j) {
                // dist += (new_image[j] - pixel[i][j]) * (new_image[j] - pixel[i][j]);
                dist += abs(new_com_image[j] - compressed_pixel[i][j]);
            }
            q.insert(make_pair(dist, i));
            while (q.size() > K) {
                pii p = *(--q.end());
                q.erase(p);
            }
        }
        valarray<int> cnt(10);
        while (!q.empty()) {
            pii p = *q.begin();
            ++cnt[label[p.second]];
            q.erase(p);
        }
        int ans = 0, mx = 0;
        for (int i = 0; i < 10; ++i) {
            if (mx < cnt[i]) ans = i, mx = cnt[i]; 
        }
        return ans;
    }
};