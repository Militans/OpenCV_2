import cv2

orb = cv2.ORB_create()

cap = cv2.VideoCapture(0)

sample_img = cv2.imread('path_image.jpg', 0)
keypoints1, descriptors1 = orb.detectAndCompute(sample_img, None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints2, descriptors2 = orb.detectAndCompute(gray_frame, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(sample_img, keypoints1, frame, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('Real-time Feature Matching', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()