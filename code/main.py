TESSERACT_PATH = '/usr/bin/tesseract'


try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2
from matplotlib import pyplot as plt
import os
from pdf2image import convert_from_path, convert_from_bytes
from PyPDF2 import PdfFileWriter, PdfFileReader
import numpy as np

def create_dir(name):
    import os
    if os.path.exists(name) == True:
        return
    os.mkdir(name)

# Первеодит изображение в gray
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def binarize(image):
    gray_image = grayscale(image)
    return cv2.threshold(gray_image, 200, 230, cv2.THRESH_BINARY)

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1,1), np.uint8)
    image = cv2.dilate(image, kernel, iterations = 1)
    kernel = np.ones((1,1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def thin_font(image): 
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def thick_font(image): 
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

# Calculate skew angle of an image
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
#     _,contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours,  hierarchy,  = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    print(angle)
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    print(angle)
    return rotateImage(cvImage, -1.0 * angle)

def remove_borders(image):
    contours, heirachy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntSorted= sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntSorted[-1]
    x, y ,w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h,x:x+w]
    
    return (crop)

def split_pdf(pdf_path, step=15):
    from PyPDF2 import PdfFileWriter, PdfFileReader
    from uuid import uuid4
    input_pdf = PdfFileReader(pdf_path)
    
    num_of_page = input_pdf.getNumPages()
    
    count_page = 0
    path_list = []
    file_name = str(uuid4())
    
    while num_of_page > 0:
        step = step if num_of_page - step >= step else num_of_page
        path = f'files/temp/{file_name}_{count_page}-{count_page + step}.pdf'
        with open(path, 'wb') as f:
            pdf_writer = PdfFileWriter()
            for page_num in range(count_page, count_page + step):
#                 print(page_num)
                pdf_writer.addPage(input_pdf.getPage(page_num))
            
            pdf_writer.write(f)
            f.close()
        path_list.append(path)
        num_of_page -= step
        count_page += step
    
    return path_list
  
def delete_path(path):
  import os
  os.remove(path)


def ocr_pdf(pdf_path):
    import os
    import pytesseract
    # text = [os.path.exists(pdf_path)]
    text = []
    # with open(pdf_path):
    #   text.append(True)
  

    # images = convert_from_path(pdf_path, output_folder="files/temp", fmt="jpg", thread_count=6, paths_only=True)
    images = convert_from_path(pdf_path, thread_count=6)
    for image in images:
#         img = cv2.imread(image)
#         i = grayscale(img)
#         _, i = binarize(i)
#         without_noise = noise_removal(i)
        without_noise = image
        text.append(pytesseract.image_to_string(without_noise, lang="rus"))
        # delete_path(image)
    # delete_path(pdf_path)
    return text
  
def ocr_img(cvImg, deskew_img=False, font_mode = '', noise_remove=False):
  img = cvImg
  # Если нужно сделать поворот изображения
  if (deskew_img):
      img = deskew(img)
  _, img = binarize(img)
  
  # Если нужно убрать шумы
  if noise_remove:
      img = noise_removal(img)
  
  # Если нужно увеличить фон
  if font_mode == 'thick':
      img = thick_font(img)

  # Если нужно уменьшить фон
  elif font_mode == 'thin':
      img = thin_font(img)

  text = pytesseract.image_to_string(img, lang="rus+eng")
  return (text, img)

    
from pyspark.sql import SparkSession

appName='tesseract-app'
master = 'local[2]'

def handle_path(path):
    from pyspark import SparkFiles
    path = SparkFiles.get(path)
    
    print(f'----- {path} ----- \n')
    data = ocr_pdf(path)
    # delete_path(path)
    return data

# def handle_path(img):
#     return img
#     data = ocr_pdf(img)
#     # delete_path(path)
#     return data

def init_spark():
  sql = SparkSession.builder\
    .appName(appName)\
    .getOrCreate()
  sc = sql.sparkContext
  return sql,sc

def get_images(paths):
  images = []
  for path in paths:
     images.append(convert_from_path(path, thread_count=6))
  return images

if __name__ == '__main__':
  _, sc= init_spark()

  files = split_pdf("files/book/rodnoe-slovo.pdf")
  for file in files:
    sc.addFile(file)

  input_files = sc.parallelize(files, 12)
  
  # input_files = sc.parallelize(split_pdf("files/book/rodnoe-slovo.pdf"), 12)
  
  converted = input_files.map(lambda x: handle_path(x))

  data = converted.collect()
  print('result !!!!!!!!!!!!!!!!!!!', data)

  # converted.coalesce(1).saveAsTextFile("file")
  sc.stop()