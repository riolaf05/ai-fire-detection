import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import Flask
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

UPLOAD_FOLDER = '/api/static/uploads/'
LABELS=['incendio', 'nessun incendio']

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def lite_model(image):
  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path="/model/lite_fire_detection_model.tflite")
  
  interpreter.allocate_tensors()
  interpreter.set_tensor(interpreter.get_input_details()[0]['index'], image)
  interpreter.invoke()
  return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		 
		image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		data = np.asarray(image, dtype="float32" )
		resized = cv2.resize(data, (224,224), interpolation=cv2.INTER_CUBIC)
		
		#inference
		probs_lite = lite_model(resized[None, ...])[0]
		predicted_index = np.argmax(probs_lite)
		flash(LABELS[predicted_index])
		
		return render_template('upload.html', filename=filename)
		
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()