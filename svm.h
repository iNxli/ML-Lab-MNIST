struct SVM {
    const double C = 45;
    const int Catagories = 10;
    vector<vector<int> > pixel;
    vector<int> label;
    vector<pair<double, vector<double> > > plain;
    vector<vector<int> > idx;
    int N = 500;

    SVM(string image_file, string label_file) {
        input_images(image_file, pixel);
        input_labels(label_file, label);
        plain.resize(Catagories * Catagories);
        idx.resize(Catagories * Catagories);
    }

    int random_index(int i, int m) {
        int j = rand() % m;
        while (i == j) j = rand() % m;
        return j;
    }

    double clip(double a, double L, double H) {
        a = a > H ? H : a;
        a = a < L ? L : a;
        return a;
    }

    double mul(const vector<int> &x, const vector<int> y) {
        double res = 0;
        for (int i = 0; i < x.size(); ++i) res += x[i] * y[i];
        return res;
    }

    void SMO(int pos, int neg, double C) {
        vector<vector<int> > x;
        vector<int> y; 
        vector<double> alpha;
        for (int i = 0; i < N; ++i) {
            if (label[i] != pos && label[i] != neg) continue;
            alpha.push_back(0);
            idx[pos * Catagories + neg].push_back(i);
            x.push_back(pixel[i]);
            y.push_back(((label[i] == pos) ? 1 : -1));
        }
        int T = 10, m = y.size();
        double b = 0;
        while (T--) {
            for (int i = 0; i < m; ++i) {
                double fxi = 0, Ei, fxj = 0, Ej;
                for (int k = 0; k < m; ++k) fxi += alpha[k] * y[k] * mul(x[i], x[k]);
                fxi += b;
                Ei = fxi - y[i];
                int j = random_index(i, m);
                for (int k = 0; k < m; ++k) fxj += alpha[k] * y[k] * mul(x[j], x[k]);
                fxj += b;
                Ej = fxj - y[j];
                double L, H;
                if (y[i] == y[j]) {
                    L = max(double(0), alpha[j] + alpha[i] - C), H = min(C, (double)(alpha[j] + alpha[i]));
                } else {
                    L = max(double(0), alpha[j] - alpha[i]), H = min(C, C + alpha[j] - alpha[i]);
                }
                double eta = mul(x[i], x[i]) + mul(x[j], x[j]) - 2.0 * mul(x[i], x[j]);
                if (fabs(eta) < 1e-6) continue;
                double aj_new = alpha[j] + 1.0 * y[j] * (Ei - Ej) / eta;
                aj_new = clip(aj_new, L, H);
                double ai_new = alpha[i] + y[i] * y[j] * (alpha[j] - aj_new);
                double b1 = b - Ei - y[i] * (ai_new - alpha[i]) * mul(x[i], x[i]) 
                    - y[j] * (aj_new - alpha[j]) * mul(x[j], x[i]);
                double b2 = b - Ej - y[j] * (ai_new - alpha[i]) * mul(x[i], x[j])
                    - y[i] * (aj_new - alpha[j]) * mul(x[j], x[j]);
                b = (b1 + b2) / 2;
            }
        } 
        plain[pos * Catagories + neg].first = b;
        plain[pos * Catagories + neg].second.resize(m, 0);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < x[i].size(); ++j) {
                plain[pos * Catagories + neg].second[j] += alpha[i] * y[i] * x[i][j];
            }
        }
    }

    void init() {
        srand((unsigned)time(0));
        for (int i = 0; i < Catagories; ++i) {
            for (int j = i + 1; j < Catagories; ++j) {
                SMO(i, j, C);
            }
        }
    }

    int recognize(int i, int j, const vector<int> &new_image) {
        double ans = 0;
        for (int k = 0; k < new_image.size(); ++k) {
            ans += new_image[k] * plain[i * Catagories + j].second[k];
        }
        return (ans + plain[i * Catagories + j].first > 0) ? i : j;
    }

};


void svm_solution(const vector<vector<int> > &test_pixel, const vector<int> &test_label) {
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
        cout << "NO. " << i << " is recognized as: " << ans << 
            " and actually is: " << test_label[i] << endl;
        _cnt += (ans == test_label[i]);
    }
    cout << "The accuracy rating is: " << setiosflags(ios::fixed) << setprecision(2) << 1.0 * _cnt / N << endl;
}