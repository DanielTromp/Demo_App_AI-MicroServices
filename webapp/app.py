# python app.py 

from celery import Celery
from celery.result import AsyncResult
from celery.utils.log import get_task_logger

import uuid
from minio import Minio
from mimetypes import MimeTypes
from tempfile import NamedTemporaryFile

from flask import Flask, redirect, render_template, request
from flask import send_from_directory, url_for
import logging

#from flask_bootstrap import Bootstrap

minio = Minio('localhost:9000',
                access_key='access_key',
                secret_key='secret_key',
                secure=False)

celery = Celery(
                broker='redis://localhost:6379/0', 
                backend='redis://localhost:6379/0'
                )

application = Flask(__name__)
logger = logging.getLogger(__name__)
celery_logger = get_task_logger(__name__)

@celery.task(name='image.processing')
def processing(filename):
    return

### Flask config ###

@application.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']

@application.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if not request.files.get('file', None):
            msg = 'the request contains no file'
            logger.error(msg)
            return render_template('exception.html', text=msg)
        
        file = request.files['file']
        if file and not allowed_file(file.filename):
            msg = f'the file {file.filename} has wrong extention'
            logger.error(msg)
            return render_template('exception.html', text=msg)

        tmp = NamedTemporaryFile()
        filename = str(uuid.uuid4()) + "_" + file.filename
        #filename = file.filename
        tmp.name = 'tmp/' + filename
        file.save(tmp.name)

        content_type = MimeTypes().guess_type(tmp.name)[0]
        try:
            minio.make_bucket('images')
        minio.fput_object('images', filename, tmp.name, 
                            content_type=content_type)

        tmp.close()
        logger.info(f'the file {tmp.name} has been successfully saved as {filename}')
        return redirect('/process/' + filename)

@application.route('/process/<filename>')
def task_processing(filename):
    task = processing.delay(filename)
    async_result = AsyncResult(id=task.task_id, app=celery)
    processing_result = async_result.get()
    return render_template("result.html", image_name=processing_result)

@application.route('/result/<filename>')
def send_image(filename):
    tmp = NamedTemporaryFile()
    tmp.name = 'tmp/' + filename
    minio.fget_object('images', 'facebox/' + filename, tmp.name)
    return send_from_directory('', tmp.name)

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)