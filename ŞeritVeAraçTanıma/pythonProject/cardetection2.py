import cv2

# Cascade yükleme (örneğin, araçlar için sınıflandırma)
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Videoyu açma
cap = cv2.VideoCapture('serit.mp4')

# Video çıkışı için VideoWriter oluşturma
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec kullanıldı
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    # Bir kare okuma
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü griye dönüştürme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Araçları tespit etme
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Tespit edilen araçları çizme
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Sonucu gösterme
    cv2.imshow('Araç Tanıma', frame)

    # Video çıkışına yazma
    out.write(frame)

    # Çıkış için 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

#videoyu kapat
cap.release()
out.release()
cv2.destroyAllWindows()
