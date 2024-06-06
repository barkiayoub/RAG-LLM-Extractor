from typing import Union

from fastapi import FastAPI
import datetime
import os
import io
import re
import csv
import json
import jsonlines
import glob
from pathlib import Path
import shutil
import random
import openai
# import librosa
import soundfile
from langdetect import detect
import pandas as pd
from tqdm import tqdm
# from IPython.display import Audio
import scipy.io.wavfile as wavfile
# from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk

SPEECH_KEY="f48272cd970c4593aeb8dce0bb7451f6"
SERVICE_REGION="francecentral"
SERVICE_ENDPOINT="https://francecentral.api.cognitive.microsoft.com"
 
OPENAI_API_TYPE="azure"
OPENAI_API_VERSION="2023-07-01-preview"

# nouveau tenant
OPENAI_API_BASE="https://openai4orange.openai.azure.com/"
OPENAI_API_KEY="25ecc25bfd9a48cdaa0982fcf20ebe5b"
DEPLOYMENT_NAME='4Orange'

# config for utils/utils_speech_text.py
SPEECH_RECOGNITION_LANG="mt-MT"
SPEECH_SYNTHESIS_LANG="mt-MT"
 

def normalize_keys(dictionary):
    normalized_dict = {}
    for key, value in dictionary.items():
        normalized_key = key.lower().replace(' ', '_')
        normalized_dict[normalized_key] = value
    return normalized_dict

def get_insights(text: list, prompt_input: str):
    for i in range(len(text)):
        text[i] = str(text[i]).replace("\n", " ")

    df = pd.DataFrame()
    df["query"] = text
    template = open(prompt_input).read().format(
        data=df[['query']].to_csv(index=False, quoting=csv.QUOTE_ALL))

    client = openai.AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY
    )
    completion = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        temperature=0,
        top_p=.5,
        messages=[
            {
                "role": "user",
                "content": template,
            },
        ],
    )
    
    # return normalize_keys(json.loads(completion.choices[0].message.content))
    return completion.choices[0].message.content

prompt_input = "/home/nairaxo/projets/awb/data/template.txt"
df = pd.read_excel("/home/nairaxo/projets/awb/data/Verbatims pour SCL.xlsx")


n = 20  #chunk row size
list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]

j = 84
for df in tqdm(list_df[84:]):
    text = df["Quelles sont vos suggestions pour améliorer nos services ?"].tolist()
    # print(text)
    try:
        text = get_insights(text, prompt_input)
    except:
        text = []
    # print(text)
    # with jsonlines.open(f"/home/nairaxo/projets/awb/data/outputs/chunk_{j}.json", "a") as writer:
    #     writer.write(text)
    f = open(f"/home/nairaxo/projets/awb/data/outputs2/chunk_{j}.txt", "w")
    f.write(text)
    j = j+1