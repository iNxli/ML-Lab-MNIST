struct Bayes {
    valarray<valarray<int> > pixel;
    valarray<int> label;
    valarray<valarray<valarray<double> > > p;
    valarray<double> q;
    const int H = 45;

    Bayes(string image_file, string label_file) { // initial function
        input_images(image_file, pixel);
        input_labels(label_file, label);
        p.resize(10);
        for (int i = 0; i < 10; ++i) p[i].resize(pixel[0].size());
        for (int i = 0; i < 10; ++i) for (int j = 0; j < p[i].size(); ++j) p[i][j].resize(2);
        q.resize(10);
    }

    void init() { // initial, to calculate p and q
        for (int i = 0; i < label.size(); ++i) q[label[i]] += 1;
        for (int i = 0; i < 10; ++i) q[i] = (q[i] + 1) / (label.size() + 2), q[i] = log(q[i]);
        valarray<int> cnt(10);
        for (int i = 0; i < label.size(); ++i) {
            for (int j = 0; j < pixel[i].size(); ++j) {
                p[label[i]][j][(pixel[i][j] > H)] += 1;
                ++cnt[label[i]];
            }
        }
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < pixel[i].size(); ++j) {
                p[i][j][0] = (p[i][j][0] + 1) / (cnt[i] + 2); // laplace smooth
                p[i][j][0] = log(p[i][j][0]);
                p[i][j][1] = (p[i][j][1] + 1) / (cnt[i] + 2); // laplace smooth
                p[i][j][1] = log(p[i][j][1]);
            }
        }
    }

    int recognize(const valarray<int> &new_image) { // predict
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