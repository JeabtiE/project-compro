import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
import imutils


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame = cv2.resize(frame, None, fx=1, fy=1)

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    result_img = closing.copy()

   
    cnts = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    hitung_objek = 0

    for cnt in cnts:
        area = cv2.contourArea(cnt)

        if area < 3000:
            continue

        hitung_objek += 1

        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        orig = frame.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (0, 255, 0), -1)

        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        lebar_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        panjang_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        cm_per_pixel = 1 / 25.5

        real_width = lebar_pixel * cm_per_pixel
        real_height = panjang_pixel * cm_per_pixel

        weight = (0.0019 * ((2 * real_width * (real_height ** 2) * 3.14) / 3)) + 0.2228

        cv2.putText(orig, f"L: {real_width:.2f} CM", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(orig, f"H: {real_height:.2f} CM", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(orig, f"Weight: {weight:.2f} kg", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 20, 127), 2)

        cv2.putText(orig, f"Objects: {hitung_objek}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 144, 255), 2)

        cv2.putText(orig, "*** Keep 60 cm distance ***",
                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        cv2.imshow("Camera", orig)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
