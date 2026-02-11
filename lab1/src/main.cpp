#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

struct BMPImage {
    BMPHeader fileHeader;
    BMPInfoHeader infoHeader;
    std::vector<uint8_t> palette;
    std::vector<uint8_t> data;
};

BMPImage readBMP(const std::string &filename) {
    BMPImage img;
    std::ifstream in(filename, std::ios::binary);

    if (!in) {
        throw std::runtime_error("Не удалось открыть BMP");
    }

    in.read((char*)&img.fileHeader, sizeof(img.fileHeader));
    in.read((char*)&img.infoHeader, sizeof(img.infoHeader));

    if (img.infoHeader.biBitCount != 8) {
        throw std::runtime_error("Только 8-битные BMP!");
    }

    img.palette.resize(256 * 4);
    in.read((char*)img.palette.data(), img.palette.size());

    int width = img.infoHeader.biWidth, height = img.infoHeader.biHeight, rowSize = ((width + 3) / 4) * 4;

    img.data.resize(rowSize * height);
    in.read((char*)img.data.data(), img.data.size());

    return img;
}

void writeBMP(const std::string &filename, const BMPImage &img) {
    std::ofstream out(filename, std::ios::binary);
    out.write((char*)&img.fileHeader, sizeof(img.fileHeader));
    out.write((char*)&img.infoHeader, sizeof(img.infoHeader));
    out.write((char*)img.palette.data(), img.palette.size());
    out.write((char*)img.data.data(), img.data.size());
}

std::vector<uint8_t> readFileBytes(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void writeFileBytes(const std::string &filename, const std::vector<uint8_t> &data) {
    std::ofstream out(filename, std::ios::binary);
    out.write((char*)data.data(), data.size());
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cout << "Использование:\n";
        std::cout << " extract_plane input.bmp output.bmp k\n";
        std::cout << " embed input.bmp message.txt output.bmp k\n";
        std::cout << " extract_msg stego.bmp output.txt k\n";
        return 0;
    }

    std::string mode = argv[1];
    int k;
    if (mode == "extract_plane" || mode == "extract_msg") {
        k = std::stoi(argv[4]) - 1;
    }
    else if (mode == "embed") {
        k = std::stoi(argv[5]) - 1;
    }

    if (k < 0 || k > 7) {
        throw std::runtime_error("k ∈ [1; 8]");  
    }

    BMPImage img = readBMP(argv[2]);

    int width = img.infoHeader.biWidth, height = img.infoHeader.biHeight, rowSize = ((width + 3) / 4) * 4;

    if (mode == "extract_plane") {
        BMPImage out = img;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * rowSize + x;
                uint8_t bit = (img.data[idx] >> k) & 1;
                out.data[idx] = bit ? 255 : 0;
            }
        }

        writeBMP(argv[3], out);
        std::cout << "Битовая плоскость сохранена в файл: " << argv[3] << '\n';
    }
    else if (mode == "embed") {
        auto msg = readFileBytes(argv[3]);

        std::vector<uint8_t> bits;
        uint32_t size = msg.size();
        for (int i = 31; i >= 0; --i) {
            bits.push_back((size >> i) & 1);
        }

        for (uint8_t c : msg) {
            for (int i = 7; i >= 0; --i) {
                bits.push_back((c >> i) & 1);
            }
        }

        int capacity = width * height;
        int written = std::min((int)bits.size(), capacity);

        int b = 0;
        for (int y = 0; y < height && b < written; ++y) {
            for (int x = 0; x < width && b < written; ++x) {
                int idx = y * rowSize + x;
                img.data[idx] &= ~(1 << k);
                img.data[idx] |= (bits[b++] << k);
            }
        }

        writeBMP(argv[4], img);
        std::cout << "Встроено бит: " << written << " в файл " << argv[4] << '\n';
    }
    else if (mode == "extract_msg") {
        std::vector<uint8_t> bits;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * rowSize + x;
                bits.push_back((img.data[idx] >> k) & 1);
            }
        }

        uint32_t size = 0;
        for (int i = 0; i < 32; ++i) {
            size = (size << 1) | bits[i];
        }

        std::vector<uint8_t> msg(size);
        int b = 32;

        for (uint32_t i = 0; i < size; ++i) {
            uint8_t c = 0;
            for (int j = 0; j < 8; ++j) {
                c = (c << 1) | bits[b++];
            }
            msg[i] = c;
        }

        writeFileBytes(argv[3], msg);
        std::cout << "Сообщение извлечено без ошибок\n";
    }
}