import logging
import keyword_extractors
import overall_sentiment
import phrase_extraction_models
import summarizer_keyphrase_extractor
from flask import Flask
from flask_jsonrpc import JSONRPC
from transcriber import transcribeVideo
from werkzeug.utils import secure_filename
from ytpull import pullYoutubeCaptions, download_video

application = Flask(__name__)
application.config.from_object('config')
jsonrpc = JSONRPC(application, '/api')
se = phrase_extraction_models.StatementExtracter()

model_dir_path = './lstm3'
#sum_key_ext = summarizer_keyphrase_extractor.SummarizerExtractor(model_dir_path)

def get_video_text_result(is_youtube, file):
	video_text = None
	try:
		if is_youtube:
			video_text = pullYoutubeCaptions(file)
		else:
			print('Begin Transcribing')
			video_text = transcribeVideo(application.config['TRANSCRIBE_JOB_NAME'], f's3://{application.config["BUCKET_NAME"]}/{file}')
			print('Finished Transcribing')
	except Exception as e:
		logging.error("Failed to get text from video: ", e)
	finally:
		return video_text

def get_video_keywords(video_text):
	keywords = None
	try:
		keywords = keyword_extractors.outputKeywords(video_text)
	except Exception as e:
		logging.error("Failed to get keywords for video: ", e)
	finally:
		return keywords

def get_isare_sentences(video_text):
    sentences = None
    try:
    	sentences = se.is_statements(video_text)
    except Exception as e:
    	logging.error("Failed to get isare sentences for video: ", e)
    finally:
    	return sentences

def get_summarizer_sentences(video_text):
    #TODO
    # Handle any exceptions
    sentences = None
    try:
    	sum_key_ext = summarizer_keyphrase_extractor.SummarizerExtractor(model_dir_path)
    	sentences = sum_key_ext.get_keyphrases(video_text)
    except Exception as e:
    	logging.error("Failed to get summarizer sentences for video: ", e)
    finally:
    	return sentences

def get_full_sentiment(video_text):
	full_sentiment = None
	try:
		sentiment = overall_sentiment.OverallSentiment(video_text)
		full_sentiment = sentiment.getSentiment(faster=True)
	except Exception as e:
		logging.error("Failed to get overall sentiment for video: ", e)
	finally:
		return full_sentiment

def get_extracted_information(video_text):
    # Get the keywords
    keywords = get_video_keywords(video_text)
    # Get the is/are sentences
    isare_sentences = get_isare_sentences(video_text)
    # Get the summarizer sentences
    summarizer_sentences = get_summarizer_sentences(video_text)
    # Get full sentiment
    full_sentiment = get_full_sentiment(video_text)

    return keywords, isare_sentences, summarizer_sentences, full_sentiment


@jsonrpc.method('get_video_text')
def get_video_text(params):
	video_text = get_video_text_result(params['is_youtube'], params['file'])
	if video_text:
		return video_text
	# TODO add captions error
	return {'error':'There was an error retrieving text for the video'}

@jsonrpc.method('get_keywords')
def get_keywords(params):
	video_text = get_video_text_result(params['is_youtube'], params['file'])
	if video_text:
		keywords = get_video_keywords(video_text)
		if keywords:
			return keywords
		return {'error':'There was an error retrieving keywords'}
	return {'error':'There was an error retrieving text for the video'}

@jsonrpc.method('get_statements')
def get_statements(params):
	video_text = get_video_text_result(params['is_youtube'], params['file'])
	if video_text:
		isare_sentences = get_isare_sentences(video_text)
		if isare_sentences:
			return isare_sentences
		return {'error':'There was an error retrieving isare phrases'}
	return {'error':'There was an error retrieving text for the video'}

@jsonrpc.method('get_summarizer_statements')
def get_summarizer_statements(params):
	video_text = get_video_text_result(params['is_youtube'], params['file'])
	if video_text:
		summarizer_sentences = get_summarizer_sentences(video_text)
		if summarizer_sentences:
			return summarizer_sentences
		return {'error':'There was an error retrieving summarizer phrases'}
	return {'error':'There was an error retrieving text for the video'}

@jsonrpc.method('get_all_information_extracted')
def get_all_information_extracted(params):
	video_text = get_video_text_result(params['is_youtube'], params['file'])
	if video_text:
		keywords, isare_sentences, summarizer_sentences, full_sentiment = get_extracted_information(video_text)
		if not keywords:
			keywords = {'error':'There was an error retrieving keywords'}
		if not isare_sentences:
			isare_sentences = {'error':'There was an error retrieving isare phrases'}
		if not summarizer_sentences:
			summarizer_sentences = {'error':'There was an error retrieving summarizer phrases'}
		if not full_sentiment:
			full_sentiment = {'error':'There was an error retrieving overall sentiment for the video'}

		dict_information = {'video_text':video_text, 'keywords':keywords, 'isare_sentences':isare_sentences, 'summarizer_sentences':summarizer_sentences, 'full_sentiment':full_sentiment}
		return dict_information
	return {'error':'There was an error retrieving text for the video'}

@jsonrpc.method('get_overall_sentiment')
def get_overall_sentiment(params):
	video_text = get_video_text_result(params['is_youtube'], params['file'])
	if video_text:
		full_sentiment = get_full_sentiment(video_text)
		if full_sentiment:
			return full_sentiment
		return {'error':'There was an error retrieving overall sentiment for the video'}
	return {'error':'There was an error retrieving text for the video'}

if __name__ == '__main__':
    application.run(port=8000, debug=True)