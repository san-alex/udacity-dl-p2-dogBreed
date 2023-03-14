from flask import Flask, redirect, request, render_template
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO

allowed_exts = {'jpg', 'jpeg','png','JPG','JPEG','PNG'}
def check_allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

def predict(model, img):
    pass

app = Flask(__name__)

@app.route('/')
def html():
    print('html')
    return render_template('index.html')

@app.route('/dldb', methods=['GET', 'POST'])
def dldb():
    '''data = request.files
    file = data.get('image')
    newFileName = file.filename.split('.')[0] + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + file.filename.split('.')[1]
    newFileName = '/static/images' + newFileName
    file.save(newFileName)
    print('data', data)
    return render_template('predict.html', img_file=newFileName)'''

    if request.method == 'POST':
        if 'image' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file and check_allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            img = Image.open(file.stream)
            with BytesIO() as buf:
                img.save(buf, 'jpeg')
                image_bytes = buf.getvalue()
            encoded_string = base64.b64encode(image_bytes).decode()

            # importing model
               
        return render_template('predict.html', img_data=encoded_string), 200
    else:
        return render_template('predict.html', img_data=""), 200


if __name__ == '__main__':
    app.run(port=5000, debug=True)