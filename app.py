from flask import Flask, render_template, request, url_for
from flask import *
import os.path
import pickle
import cv2
from skimage import feature


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/info')
def information():
    return render_template('info.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

# @app.route('/predict' ,methods=['POST','GET'])
# def predict():
#     return 'Hello World'


@app.route('/predict', methods=['POST']) 
def predict():
    if request.method == 'POST':
        f=request.files['file'] #requesting the file
        basepath=os.path.dirname(__file__)#storing the file directory
        filepath=os.path.join(basepath, "./uploads", f.filename)#storing the file in uploads folder 
        f.save(filepath) #saving the file
        #Loading the saved model 
        print("[INFO] Loading model...") 
        model = pickle.loads(open('Training\parkinson.pkl', "rb").read())
        # pre-process the image in the same manner we did earlier image 
        image = cv2.imread(filepath) 
        output = image.copy()
        # load the input image, convert it to grayscale, and resize 
        output = cv2.resize(output, (128, 128)) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200)) 
        image = cv2.threshold(image, 0, 255, 
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # quantify the image and make predictions based on the extracted
        # features using the last trained Random Forest
        features = feature.hog(image, orientations=9, pixels_per_cell= (10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1") 
        preds = model.predict([features])
        print(preds)
        ls=["healthy", "parkinson"]
        result = ls[preds[0]]
        # draw the colored class label on the output image and add it to 
        # # the set of output images
        color = (0, 255, 0) if result == "healthy" else (0, 0, 255)
        cv2.putText(output, result, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 
        # cv2.imshow("Output", output) 
        # cv2.waitKey(0) 
        if(result == 'healthy'):
            return render_template('upload.html',dis='Hurrayy you are Healthy!!!')
        else:
            return render_template('upload.html',dis="Ohh No!!! You are affected by Parkinson's disease.. Don't Worry we'll cure it!! ")
        return result 
    return None



if __name__ == "__main__":
    app.run(debug=True)
