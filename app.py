import cv2 as cv

src = cv.imread('q10.jpg', cv.IMREAD_COLOR)
gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
cv.imwrite("debug/binary.jpg", binary)


dst, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
dst = cv.imread('q10.jpg', cv.IMREAD_COLOR)
dst = cv.drawContours(dst, contours, -1, (0, 0, 255, 255), 2, cv.LINE_AA)
cv.imwrite('debug/debug_1.jpg', dst)


th_area = binary.shape[0] * binary.shape[1] / 100
contours_large = list(filter(lambda c:cv.contourArea(c) > th_area, contours))

dst = cv.imread('q10.jpg', cv.IMREAD_COLOR)
for (i,cnt) in enumerate(contours_large):
    arclen = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02*arclen, True)
    if len(approx) < 4:
        continue
    dst = cv.drawContours(dst, [approx], -1, (0, 0, 255, 255), 2, cv.LINE_AA)

cv.imwrite('debug/debug_2.jpg', dst)
