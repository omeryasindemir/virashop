from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # CORS'u içe aktar
import cv2
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimum 16 MB
CORS(app)  # Bu satır tüm orijinlere izin verir


@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Kullanıcıdan gelen görselleri al
        background_file = request.files['background']
        pattern_file = request.files['pattern']

        # Gelen dosyaları kaydet
        background_path = "background.jpg"
        pattern_path = "pattern.jpg"
        background_file.save(background_path)
        pattern_file.save(pattern_path)

        # Görselleri OpenCV ile yükle
        background = cv2.imread(background_path)
        pattern = cv2.imread(pattern_path)

        # Görsel işleme (Yeşil alanı tespit etme)
        hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])  # Yeşil renk alt sınırı
        upper_green = np.array([80, 255, 255])  # Yeşil renk üst sınırı
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Yeşil alanın koordinatlarını bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return jsonify({"error": "Yeşil alan bulunamadı"}), 400

        # En büyük konturu al
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) != 4:
            return jsonify({"error": "Yeşil alan dörtgen değil"}), 400

        # Perspektif dönüşüm için hedef noktalar
        target_points = np.float32([[0, 0], [pattern.shape[1], 0], [pattern.shape[1], pattern.shape[0]], [0, pattern.shape[0]]])
        source_points = np.float32([point[0] for point in approx])

        # Perspektif dönüşüm matrisini hesapla
        matrix = cv2.getPerspectiveTransform(target_points, source_points)
        warped_pattern = cv2.warpPerspective(pattern, matrix, (background.shape[1], background.shape[0]))

        # Halıyı yeşil alan üzerine yerleştir
        mask_inv = cv2.bitwise_not(mask)
        background_no_green = cv2.bitwise_and(background, background, mask=mask_inv)
        result = cv2.add(background_no_green, warped_pattern)

        # Sonucu kaydet
        output_path = "output.jpg"
        cv2.imwrite(output_path, result)

        # İşlenmiş görseli geri döndür
        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
