//========================= auxiliary function

uint32_t endian_transit(uint32_t x) {
    x = ((x << 8) & 0xff00ff00) | ((x >> 8) & 0x00ff00ff);
    return (x << 16) | (x >> 16);
}

void read(uint32_t *x, ifstream &IN) {
    IN.read((char *)(x), 4);
    *x = endian_transit(*x);
}

//========================== input

void input_images(string file, valarray<valarray<int> > &pixel) {
    ifstream IN(file, ios::binary);
    uint32_t magic_number;
    uint32_t image_number;
    uint32_t row_number;
    uint32_t col_number;
    read(&magic_number, IN);
    read(&image_number, IN);
    read(&row_number, IN);
    read(&col_number, IN);
    valarray<int> a;
    pixel.resize(image_number);
    for (int i = 0; i < image_number; ++i) pixel[i].resize(row_number * col_number);
    for (int i = 0; i < image_number; ++i) {
        for (int j = 0; j < row_number * col_number; ++j) {
            IN.read((char *)(&pixel[i][j]), 1);
        }
    }
}

void input_labels(string file, valarray<int> &label) {
    ifstream IN(file, ios::binary);
    uint32_t magic_number;
    uint32_t item_number;
    read(&magic_number, IN);
    read(&item_number, IN);
    label.resize(item_number);
    for (int i = 0; i < item_number; ++i) IN.read((char *)(&label[i]), 1);
}