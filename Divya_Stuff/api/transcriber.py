from urllib.request import urlopen
import time
import boto3
import json

def transcribeVideo(job_name, s3_uri):
    """
    job_name:
    s3_uri:'s3://topic-detection-raw-data/test.mp4'
    """
    transcribe = boto3.client('transcribe', region_name='us-west-2')
    transcribe.start_transcription_job(TranscriptionJobName=job_name,
                                       Media={'MediaFileUri': s3_uri}, 
                                       MediaFormat='mp4',
                                       LanguageCode='en-US')
    
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Not ready yet...")
        time.sleep(60)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'FAILED':
        return #todo- should i throw exception?
    
    json_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
    json_url = urlopen(json_url)
    transcription_json = json.loads(json_url.read())
    transcription_str = transcription_json['results']['transcripts'][0]['transcript']
    
    transcribe.delete_transcription_job(TranscriptionJobName=job_name) # we won't need the job again right? safe to delete?
    
    return transcription_str