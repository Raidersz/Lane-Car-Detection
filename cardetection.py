import cv2


class cardetection:
    def detect(im):

        #im = cv2.imread(img)
        car_cascade = cv2.CascadeClassifier('cars.xml')
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 9)
    # if str(np.array(cars).shape[0]) == '1':
    #     i += 1
    #     continue
        for (x,y,w,h) in cars:
           plate = im[y:y + h, x:x + w]
           cv2.rectangle(im,(x,y),(x +w, y +h) ,(51 ,51,255),2)
           cv2.rectangle(im, (x, y - 40), (x + w, y), (51,51,255), -2)
           cv2.putText(im, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return im