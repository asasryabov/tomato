from pioneer_sdk import Pioneer, Camera
import cv2
import numpy as np
import time
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageDraw

image_size = 224
model_path = "./checkpoint-.87"
model = load_model(model_path)

pm = Pioneer()
camera = Camera()

def predict(img):
    frame = camera.get_cv_frame()
    pil_img = Image.fromarray(frame)
    pil_resized = pil_img.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    img_tensor = image.img_to_array(pil_resized)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    predicted_result = model.predict(img_tensor)
    category_idx = predicted_result[0].argmax()
    cv2.imshow("1", frame)
    return category_idx

if __name__ == "__main__":
    try:
        category_idx = predict()
        print("******#####@@@@@@@@@@@@******* ", category_idx)
        cv2.waitKey(0)

    finally:
        print("*************************** FAIL")
        time.sleep(1)

        pm.land()

        pm.close_connection()
        del pm
