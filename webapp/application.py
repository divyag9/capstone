import boto3
import os
import rake
from flask import Flask, render_template, request
from ytpull import pullYoutubeCaptions, download_video


application = Flask(__name__)
# should we put in app.config? what will be difference?


@application.route("/")
def home():
    return render_template("index.html")
    
@application.route("/getkeywords_youtube", methods=['POST'])
def getkeywords_youtube():
    url = request.form['url']
    print('url:',url)
    video_text = pullYoutubeCaptions(url)
    r = rake.Rake(use_POS=True)
    r.extract_keywords(video_text)
    keywords = r.get_key_words(n=25)
    #download_video(url) #handle exceptions
    return render_template("index.html", video_text=video_text, keywords=keywords, video_available=True) # do we have to use redirect?
    
if __name__ == "__main__":
    application.run(debug=True)