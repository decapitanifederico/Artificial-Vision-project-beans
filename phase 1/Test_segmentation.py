#%%
import numpy as np  # the numpy module is associated with the alias np
import cv2          # OpenCV module
import matplotlib
import matplotlib.pyplot as plt    # the matplotlib.pyplot with the alias plt
from IPython.display import Image, display, clear_output  # Shows an image in the notebook
import time                        # module to mesuare time cost
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)


#%%
print('Numpy:',np.__version__)
print('OpenCv:',cv2.__version__)


# %%
#Pre-Processing
#Load the Image
beans_img = cv2.imread('../imgs/Beans.bmp')
beans_img = cv2.cvtColor(beans_img, cv2.COLOR_BGR2RGB)

# Display using Matplotlib
plt.imshow(beans_img)
plt.axis('off')  # Hide axes
plt.show()



# %%
img_gray = cv2.cvtColor(beans_img, cv2.COLOR_BGR2GRAY)
# %%
# Shows the processed image
Image(cv2.imencode('.jpg', img_gray)[1])

# %%
#Shows the histogram
plt.hist(img_gray.ravel(),256,[0,256], color = "gray")
plt.show()

#%%
height, width = img_gray.shape

border_x = int(height * 0.02)  
border_y = int(width * 0.02)   

# Berechne neue Begrenzungen
x_ini = border_x
x_end = height - border_x
y_ini = border_y
y_end = width - border_y

# Ausschneiden des zentralen Bereichs ohne Ränder
img_cropped = img_gray[x_ini:x_end, y_ini:y_end]
print('Cropped region of interest:', x_ini, x_end, y_ini, y_end)
img_resized = img_gray[x_ini:x_end, y_ini:y_end]

# Shows cropped image
Image(cv2.imencode('.jpg', img_resized)[1])

# %%
# Fixed threshold method
start_time = time.time()
threshold = 75
ret, img_segmented = cv2.threshold(img_resized, threshold, 255, cv2.THRESH_BINARY_INV)
print("--- %s seconds ---" % (time.time() - start_time))

# Shows segmented image
Image(cv2.imencode('.jpg', img_segmented)[1])



# %%
# Application of the open transformation with a (6.6) pixel kernel 
start_time = time.time()
kernel = np.ones((6,6),np.uint8)
img_opened = cv2.morphologyEx(img_segmented, cv2.MORPH_OPEN, kernel)
print("--- %s seconds ---" % (time.time() - start_time))

# Mostrar la imagen procesada
Image(cv2.imencode('.jpg', img_opened)[1])


# %%
# Application of the close transformation with a (6.6) pixel kernel 
start_time = time.time()
kernel = np.ones((6,6),np.uint8)
img_opened_closed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel)
print("--- %s seconds ---" % (time.time() - start_time))

# Mostrar la imagen procesada
Image(cv2.imencode('.jpg', img_opened_closed)[1])

# %%

# In OpenCv, the labeling algorithm is called findContours
contours, hierarchy = cv2.findContours(img_opened_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# All the contours of the objects of interest are drawn in green with a thickness of 3
img_contours =beans_img.copy()
img_contours = beans_img.copy()
img_cropped_contours = img_contours[x_ini:x_end, y_ini:y_end]

cv2.drawContours(img_cropped_contours, contours, -1, (0,255,0), 3)

# Shows image with contours
Image(cv2.imencode('.jpg', img_cropped_contours)[1])

# %%
start_time = time.time()
for cnt in contours:
    
    #display_handle=display(None, display_id=True)
    # Copy the original image
    img_contours = beans_img.copy()
    img_cropped_contours = img_contours[x_ini:x_end, y_ini:y_end]

    # Draws each contour
    cv2.drawContours(img_cropped_contours, [cnt], 0, (0,255,0), 3)

    # Shows each contour
    #display_handle.update(Image(cv2.imencode('.jpg', img_contours)[1]))
    
    #input('Press enter to continue...')
    #clear_output()

    #display_handle.update(None)

print('Number of Elements:', len(contours))
print("--- %s seconds ---" % (time.time() - start_time))
# %%

