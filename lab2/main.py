import numpy as np
import cv2
from math import log10


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    return img


def save_image(path: str, img: np.ndarray):
    cv2.imwrite(path, img)


def calculate_psnr(original: np.ndarray, stego: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float32) - stego.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    return 10 * log10((255 ** 2) / mse)


def prepare_watermark(path: str, size=(350, 350)):
    wm = load_image(path)
    wm = cv2.resize(wm, size)
    _, wm_bin = cv2.threshold(wm, 127, 1, cv2.THRESH_BINARY)
    return wm_bin


def embed_lsb(container: np.ndarray, watermark_bits: np.ndarray, key: int):
    stego = container.copy()
    flat = stego.flatten()

    np.random.seed(key)
    indices = np.random.permutation(len(flat))

    for i, bit in enumerate(watermark_bits):
        flat[indices[i]] = (flat[indices[i]] & 0xFE) | bit

    return flat.reshape(container.shape)


def extract_lsb(stego: np.ndarray, num_bits: int, key: int):
    flat = stego.flatten()

    np.random.seed(key)
    indices = np.random.permutation(len(flat))

    bits = np.zeros(num_bits, dtype=np.uint8)

    for i in range(num_bits):
        bits[i] = flat[indices[i]] & 1

    return bits


def compute_available_adaptive_pixels(container: np.ndarray, T: int):
    h, w = container.shape
    count = 0

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            if (i + j) % 2 != 0:
                continue

            window = container[i - 1:i + 2, j - 1:j + 2]
            contrast = int(window.max()) - int(window.min())

            if contrast > T:
                count += 1

    return count


def embed_adaptive(container: np.ndarray,
                   watermark_bits: np.ndarray,
                   T: int):

    stego = container.copy()
    h, w = container.shape
    bit_idx = 0
    total_bits = len(watermark_bits)

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            if (i + j) % 2 != 0:
                continue

            if bit_idx >= total_bits:
                return stego

            window = container[i - 1:i + 2, j - 1:j + 2]
            contrast = int(window.max()) - int(window.min())

            if contrast > T:
                stego[i, j] = (stego[i, j] & 0xFE) | watermark_bits[bit_idx]
                bit_idx += 1

    raise ValueError("Недостаточно подходящих пикселей для внедрения.")


def extract_adaptive(stego: np.ndarray,
                     num_bits: int,
                     T: int):

    h, w = stego.shape
    bits = []

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            if (i + j) % 2 != 0:
                continue

            if len(bits) >= num_bits:
                return np.array(bits, dtype=np.uint8)

            window = stego[i - 1:i + 2, j - 1:j + 2]
            contrast = int(window.max()) - int(window.min())

            if contrast > T:
                bits.append(stego[i, j] & 1)

    raise ValueError("Недостаточно бит для извлечения.")

def run(container_path, watermark_path, method="lsb", key=42, T=10):

    print("\n==============================")
    print("Метод:", method)
    print("==============================")

    container = load_image(container_path)

    if container.shape != (512, 512):
        raise ValueError("Контейнер должен быть 512×512")

    #watermark_bin = prepare_watermark(watermark_path, size=(198, 200))
    if method == "adaptive":
        watermark_bin = prepare_watermark(watermark_path, size=(198, 200))
    else:
        watermark_bin = prepare_watermark(watermark_path, size=(400, 400))
    wm_bits = watermark_bin.flatten()

    if method == "lsb":
        available_capacity = container.size

    elif method == "adaptive":
        available_capacity = compute_available_adaptive_pixels(container, T)

    else:
        raise ValueError("Метод должен быть 'lsb' или 'adaptive'")

    print(f"Доступная емкость метода: {available_capacity}")
    print(f"Размер watermark: {len(wm_bits)}")

    if len(wm_bits) < available_capacity // 2:
        raise ValueError("Объем ЦВЗ должен быть ≥ 50% доступной емкости метода")

    if len(wm_bits) > available_capacity:
        raise ValueError("Watermark превышает доступную емкость")

    if method == "lsb":
        stego = embed_lsb(container, wm_bits, key)
        extracted_bits = extract_lsb(stego, len(wm_bits), key)

    elif method == "adaptive":
        stego = embed_adaptive(container, wm_bits, T)
        extracted_bits = extract_adaptive(stego, len(wm_bits), T)

    extracted = extracted_bits.reshape(watermark_bin.shape) * 255
    extracted = extracted.astype(np.uint8)

    psnr_container = calculate_psnr(container, stego)

    print(f"PSNR: {psnr_container:.2f} dB")

    save_image("../output/stego.bmp", stego)
    save_image("../output/extracted.bmp", extracted)

    return psnr_container

if __name__ == "__main__":

    container_path = "../fileImages/containers/1000/4.bmp"
    watermark_path = "../fileImages/logo/logo1.jpg"

    mode = input("Enter mode (lsb/adaptive): ")

    if mode == "lsb":
        run(container_path,
            watermark_path,
            method="lsb",
            key=np.random.randint(1, 101))

    elif mode == "adaptive":
        run(container_path,
            watermark_path,
            method="adaptive",
            T=10)

    else:
        print("Unknown mode")