import youtube_dl
import re
import os
import numpy as np

# Logger for youtube-dl
class MyLogger(object):

    def __init__(self):
        self.text = []
        
    def debug(self, msg):
        self.text.append(msg)

    def warning(self, msg):
        pass

    def error(self, msg):
        self.text.append(msg)

def pullYoutubeCaptions(URL):
    logger = MyLogger()
    
    # Settings that feed into the YoutubeDL object that downloads ONLY the captions
    ydl_opts = {
        'outtmpl' : 'Captions',
        'quiet' : True,
        'forcetitle' : True,
        'writesubtitles' : True,
        'writeautomaticsub' : True,
        #'listsubtitles' : True,
        #'allsubtitles' : True,
        'subtitleslangs': ['en'],
        'skip_download' : True,
        'logger' : logger
    }
    
    # Download the captions file to the local directory
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([URL])
        
    # Pull the title of the video from the Logger
    title = ""
    for text in logger.text:
        if text[0:9] != '[youtube]':
            if text[0:6] != '[info]':
                title = text
    print(title)
        
    # If we successfully download the captions file
    try:    
        # Open the downloaded captions file, extract the text, then close the file
        file = open('Captions.en.vtt','r')
        captions = file.readlines()
        file.close()
        
        # Use RegEx to remove timestamps and convert the captions to a paragraph of text.
        subtitles = np.array([])
        for line in captions:
            line = re.sub(r'WEBVTT\n','',line)
            line = re.sub(r'Kind: captions\n','',line)
            line = re.sub(r'Language: en\n','',line)
            line = re.sub(r'\d\d:\d\d:\d\d.\d\d\d --> \d\d:\d\d:\d\d.\d\d\d\n', '', line)
            line = re.sub(r'\d\d:\d\d:\d\d.\d\d\d --> \d\d:\d\d:\d\d.\d\d\d align:start position:\d+%\n','',line)
            line = re.sub(r'<\d+:\d+:\d+.\d+><c>','',line)
            line = re.sub(r'</c>', '', line)
            line = re.sub(r'(\n)', '', line)
            if (line != '') and (line != ' '):
                if line not in list(subtitles):
                    subtitles = np.append(subtitles, line)
        captions = ''
        for i in subtitles:
            captions = captions + ' ' + i
        
        # Now that we're done with the captions file, delete it from the directory
        os.remove('Captions.en.vtt')
    
    # If there is no captions file to download
    except FileNotFoundError:
        captions = ""

    return captions