import boto3
import os
import rake
import logging
import requests
from werkzeug.utils import secure_filename
from transcriber import transcribeVideo
from flask import Flask, render_template, request
from flask_jsonrpc.proxy import ServiceProxy


application = Flask(__name__)
application.config.from_object('config')

def get_parsed_information(response):
    video_text, keywords, isare_sentences, summarizer_sentences, video_text_sentences, keyword_sentences, full_sentiment, error  = None, None, None, None, None, None, None, None
    if 'error' not in response['result']:
        video_text, keywords, isare_sentences, summarizer_sentences, full_sentiment = response['result']['video_text'], response['result']['keywords'], response['result']['isare_sentences'], response['result']['summarizer_sentences'], response['result']['full_sentiment']
        if 'error' not in keywords:
            keyword_sentences = {}
            for _, sentiment, sentence in keywords[0]:
                keyword_sentences[sentence] = sentiment
            video_text_sentences = keywords[1]
            keywords = keywords[0]
        else:
            error = keywords['error']
        if 'error' in video_text:
            error = video_text['error']
        if 'error' in isare_sentences:
            error = isare_sentences['error']
        if 'error' in summarizer_sentences:
            error = summarizer_sentences['error']
        if 'error' in full_sentiment:
            error = full_sentiment['error']
    else:
        error = response['result']['error']

    return video_text, keywords, isare_sentences, summarizer_sentences, video_text_sentences, keyword_sentences, full_sentiment, error

def upload_file_to_s3(file, bucket_name):
    s3 = boto3.client('s3',
                     aws_access_key_id=application.config['AWS_ACCESS_KEY_ID'],
                     aws_secret_access_key=application.config['AWS_SECRET_ACCESS_KEY'])
    try:
        s3.upload_fileobj(
            file,
            bucket_name,
            file.filename
        ) # check condition where two people are uploading a file with same name how to avoid that
    except Exception as e:
        logging.error("Upload to s3 failed: ", e)
        return

    return file.filename


@application.route("/")
def home():
    return render_template("index.html")

@application.route("/get_information_youtube", methods=['POST','GET'])
def get_information_youtube():
    url = request.form['url']
    if url:
        server = ServiceProxy('http://api-video.5tc9jaytac.us-west-2.elasticbeanstalk.com/api')
        response = server.get_all_information_extracted({'is_youtube':True, 'file':url})
        video_text, keywords, isare_sentences, summarizer_sentences, video_text_sentences, keyword_sentences, full_sentiment, error = get_parsed_information(response)
        return render_template("index.html", video_text=video_text, keywords=keywords, isare_sentences=isare_sentences, summarizer_sentences=summarizer_sentences, video_available=True, video_text_sentences=video_text_sentences, keyword_sentences=keyword_sentences, full_sentiment=full_sentiment, error=error)

    return render_template("index.html", error="Enter a valid url")

@application.route("/get_information_upload", methods=['POST','GET'])
def get_information_upload():
    if "user_file" not in request.files:
        return render_template("index.html", error="No user_file key in request.files")
    file = request.files["user_file"]
    # There is no file selected to upload
    if file and file.filename == "":
        return render_template("index.html", error="Please select a file")
    # File is selected, upload to S3 and show S3 URL
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in application.config['ALLOWED_EXTENSIONS']:
        file.filename = secure_filename(file.filename)
        uploaded_file_name = upload_file_to_s3(file, application.config["BUCKET_NAME"])
        if uploaded_file_name:
            server = ServiceProxy('http://api-video.5tc9jaytac.us-west-2.elasticbeanstalk.com/api')
            response = server.get_all_information_extracted({'is_youtube':False, 'file':uploaded_file_name})
            video_text, keywords, isare_sentences, summarizer_sentences, video_text_sentences, keyword_sentences, full_sentiment, error = get_parsed_information(response)
            return render_template("index.html", video_text=video_text, keywords=keywords, isare_sentences=isare_sentences, summarizer_sentences=summarizer_sentences, video_available=True, video_text_sentences=video_text_sentences, keyword_sentences=keyword_sentences, full_sentiment=full_sentiment, error=error)
        return render_template("index.html", error="Error occured while uploading the video to s3")
    return render_template("index.html", error="Error with video file, check the extension")


@application.route('/data', methods=['GET'])
def data_func():
    return render_template('data.html')

@application.route('/model', methods=['GET'])
def model_func():
    return render_template('model.html')

@application.route('/about', methods=['GET'])
def about_func():
    return render_template('about.html')


if __name__ == "__main__":
    application.run(port=8080, debug=True)