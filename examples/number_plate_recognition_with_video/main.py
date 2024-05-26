import cv2
from util.ocr import OCR
from util.Platefinder import PlateFinder
from util.util import *

if __name__ == "__main__":
    findPlate = PlateFinder(minPlateArea=4100, maxPlateArea=15000)
    model = OCR(
        modelFile="model/binary_128_0.50_ver3.pb",
        labelFile="model/binary_128_0.50_labels_ver2.txt",
    )
    cap = cv2.VideoCapture("test.MOV")
    while cap.isOpened():
        ret, img = cap.read()

        if ret == True:
            cv2.imshow("original video", img)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

            possible_plates = findPlate.find_possible_plates(img)
            if possible_plates is not None:
                for i, p in enumerate(possible_plates):
                    chars_on_plate = findPlate.char_on_plate[i]
                    recognized_plate, _ = model.label_image_list(
                        chars_on_plate, imageSizeOuput=128
                    )

                    print(recognized_plate)
                    cv2.imshow("plate", p)

                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
