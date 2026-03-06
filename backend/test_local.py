import cv2
from forensic_engine import ForensicEngine

img = cv2.imread('lena.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

engine = ForensicEngine()
res = engine._analyze_faces(img_rgb)
print("LOCAL CLASS OUTPUT:", res)
