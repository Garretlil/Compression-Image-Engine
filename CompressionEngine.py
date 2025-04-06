import heapq
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#точность квантования
QUANT_BITS = 7
QUANT_LEVELS = 2 ** QUANT_BITS
MIN_MULT = 0.5
MAX_MULT = 2.0
EPSILON = 1e-6

def quantize_multiplier(m):
    m_clamped = np.clip(m, MIN_MULT, MAX_MULT)
    idx = ((m_clamped - MIN_MULT) / (MAX_MULT - MIN_MULT) * (QUANT_LEVELS - 1)).round().astype(np.uint8)
    return idx

def dequantize_multiplier(idx):
    return MIN_MULT + (idx / (QUANT_LEVELS - 1)) * (MAX_MULT - MIN_MULT)

def predict_pixel(i, j, image):
    h, w, _ = image.shape
    neighbors = []
    if i > 0:
        neighbors.append(image[i - 1, j])
    if j > 0:
        neighbors.append(image[i, j - 1])
    if neighbors:
        return np.mean(neighbors, axis=0)
    else:
        return image[i, j]

def encode_mpc(image):
    """Кодирование изображения с помощью MPC с разницей и Хаффманом"""
    h, w, c = image.shape
    differences = np.zeros_like(image, dtype=np.int32)
    
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                continue
            pred = predict_pixel(i, j, image)
            diff = image[i, j] - pred
            differences[i, j] = diff
    
    flattened_diffs = differences.flatten()
    return flattened_diffs

def huffman_encoding(data):
    """Реализация энкодинга Хаффмана для сжатия"""
    freq = {}
    for value in data:
        if value not in freq:
            freq[value] = 0
        freq[value] += 1
    
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    huff = sorted(heap[0][1:], key=lambda p: (len(p[-1]), p))
    huff_dict = {symbol: code for symbol, code in huff}
    
    encoded_data = ''.join(huff_dict[symbol] for symbol in data)
    return huff_dict, encoded_data

def decode_huffman(encoded_data, huff_dict, shape):
    """Декодирование данных с помощью Хаффмана"""
    reverse_dict = {code: symbol for symbol, code in huff_dict.items()}
    current_code = ''
    decoded_data = []
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_dict:
            decoded_data.append(reverse_dict[current_code])
            current_code = ''
    
    decoded_data = np.array(decoded_data)
    return decoded_data.reshape(shape)

img = Image.open("testImage.jpg").convert("RGB")
img_np = np.array(img)

flattened_diffs = encode_mpc(img_np)
huff_dict, encoded_data = huffman_encoding(flattened_diffs)

compressed_size = len(encoded_data) / 8 

original_size = img_np.nbytes 

decoded_diffs = decode_huffman(encoded_data, huff_dict, img_np.shape)

reconstructed_img = np.zeros_like(img_np, dtype=np.float32)
reconstructed_img[0, 0] = img_np[0, 0] 

h, w, c = img_np.shape
for i in range(1, h):
    for j in range(1, w):
        pred = predict_pixel(i, j, reconstructed_img)
        reconstructed_img[i, j] = pred + decoded_diffs[i, j]

reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)
print(f"размер исходного изображения: {original_size / 1024:.2f} KB")
print(f"размер сжатого изображения: {compressed_size / 1024:.2f} KB")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_np)
axes[0].set_title("оригинал")
axes[0].axis("off")
axes[1].imshow(reconstructed_img)
axes[1].set_title("восстановлено (MPC + Хаффман)")
axes[1].axis("off")
axes[2].imshow(np.abs(img_np - reconstructed_img))
axes[2].set_title("ошибки восстановления")
axes[2].axis("off")
plt.show()
