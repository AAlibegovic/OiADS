import cv2 as cv
import numpy as np
import datetime
import os
from collections import deque

# Provjeri da li postoji direktorijum 'detected_videos', i ako ne postoji, kreiraj ga
if not os.path.exists('detected_videos'):
    os.makedirs('detected_videos')

class Camera:
    # Učitaj pretreniran model iz Caffe formata sa zadatim konfiguracionim fajlom i težinama modela
    net = cv.dnn.readNetFromCaffe('models/config.txt', 'models/mobilenet_iter_73000.caffemodel')
    cap = cv.VideoCapture(0)
    
    # Inicijalno nije postavljen izlaz za video zapis
    out = None
    
    # Definiši trajanje međuspremnika u sekundama
    buffer_duration = 5  
    


    def __init__(self):
        # Postavi brzinu kadrova za kameru
        self.frame_rate = (self.cap.get(cv.CAP_PROP_FPS)*(1/2))
        # Izračunaj veličinu međuspremnika na osnovu brzine kadrova i trajanja međuspremnika
        self.buffer_size = int(self.frame_rate * self.buffer_duration)
        # Inicijalizuj deque (dvosmjernu listu) sa maksimalnom dužinom da bi veličina međuspremnika bila fiksna
        self.buffer = deque(maxlen=self.buffer_size)


    def run(self):
        print("Camera started...")
        while True:
            # Čitaj trenutni kadar sa kamere
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Pretvori kadar u blob format za unos u neuralnu mrežu
            blob = cv.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            
            # Postavi blob kao unos u mrežu
            self.net.setInput(blob)

            # Izvrši predikciju koristeći mrežu
            detections = self.net.forward()
            person_detected = False

            # Prođi kroz sve detekcije
            for i in range(detections.shape[2]):
                # Dobavi pouzdanost detekcije (konfidenciju)
                confidence = detections[0, 0, i, 2]
                
                # Dobavi indeks klase detektovanog objekta
                idx = int(detections[0, 0, i, 1])
                
                # Ako je detektovan objekat klase '15' (osoba) i konfidencija je veća od 0.5
                if idx == 15 and confidence > 0.5:
                    # Izračunaj koordinate pravougaonika oko detektovane osobe
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Nacrtaj pravougaonik oko detektovane osobe
                    cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    
                    # Označi da je osoba detektovana
                    person_detected = True

            # Dodaj rezultat detekcije (True/False) u međuspremnik
            self.buffer.append(person_detected)

            # Provjeri da li je osoba detektovana u bilo kojem od nedavnih kadrova
            if sum(self.buffer) > 0:  
                # Ako nije postavljen izlaz za video, postavi ga
                if self.out is None:
                    # Dobavi trenutni datum i vrijeme i formatiraj ih za ime fajla
                    now = datetime.datetime.now()
                    formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")
                    print("Person motion detected at", formatted_now)
                    
                    # Kreiraj naziv za snimljeni video fajl
                    current_recording_name = os.path.join('detected_videos', f'{formatted_now}.mp4')
                    fourcc = cv.VideoWriter_fourcc(*'mp4v')
                    # Inicijalizuj video zapisnik sa zadatom brzinom kadrova i veličinom kadra
                    self.out = cv.VideoWriter(current_recording_name, fourcc, self.frame_rate, (frame.shape[1], frame.shape[0]))
                # Zapiši trenutni kadar u video fajl
                self.out.write(frame)
            else:
                # Ako više nema detekcije, oslobodi resurse video zapisnika
                if self.out is not None:
                    self.out.release()
                    self.out = None
                    print("Recording stopped")
            
            # Prikazivanje trenutnog kadra u prozoru
            cv.imshow('Camera Feed', frame)

            # Ako je pritisnuto dugme 'q', prekini petlju
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Oslobodi resurse video zapisnika ako su još uvek zauzeti
        if self.out is not None:
            self.out.release()
            self.out = None
            
        # Oslobodi resurse kamere i zatvori sve OpenCV prozore
        self.cap.release()
        cv.destroyAllWindows()
        print("Camera released...")



def main():
    # Kreiraj instancu klase Camera i pokreni metodu run
    camera = Camera()
    camera.run()

if __name__ == "__main__":
    main()

