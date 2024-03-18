'''

'''
import openai
from openai import OpenAI
import base64
import requests
import json
import collections
import os
import sys
import time
from PIL import Image
import cv2
import numpy as np                       

def remove_unicode(text):
    # Create an empty string to store the cleaned text
    cleaned_text = ''
    
    for char in text:
        if char < 128:  # Check if the character is ASCII
            cleaned_text += chr(char)
        elif len(cleaned_text) > 0 and cleaned_text[-1] != ' ':
            cleaned_text += ' '
    return cleaned_text

def str2json(string):
    json_str = ''
    s_ind, e_ind = 0, len(string)
    
    for i, ch in enumerate(string):
        if ch == '{' or ch == '[':
            s_ind = i+1
            break
    for i, ch in enumerate(string[::-1]):
        if ch == '}'or ch == ']':
            e_ind = len(string) - i - 1
            break
#     json_str = string[s_ind:e_ind].encode("ascii", "ignore").decode()
    json_str = string[s_ind:e_ind]

    dic = {}
    for raw_item in json_str.split('},'):
        item = raw_item.strip()
        if "}" != item.strip()[-1]:
            dict_str = item.strip()+'}'
        else:
            dict_str = item.strip()
        temp_dict = eval(dict_str)
        try:
            temp_dict = eval(dict_str)
        except NameError as e:
            raise NameError
        try:
            dic[temp_dict['symbol name']] = temp_dict['description']
        except KeyError as e:
            raise KeyError
    return dic

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')  

def _call_gpt_api(image_path, cur_attempt):
    client = OpenAI()
    base64_image = encode_image(image_path)
    try:
        if cur_attempt >= 8: # send simpler prompt if GPT4 doesn't give good response after seven attempts
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                {
                  "role": "user",
                  "content": [
                    {"type": "text", "text": "Please extract information in the image? Please display the response as a list of dictionaries, the dictionary has two keys: symbol name and description. Please make sure the description is completed. The results should only have the dictionaries."},
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                      },
                    },
                  ],
                }
              ],
              max_tokens=4096,
            )

        else:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                {
                  "role": "user",
                  "content": [
                    {"type": "text", "text": "Please extract all symbols and corresponding descriptions in the image? Please display the response as a list of dictionaries, the dictionary has two keys: symbol name and description. The \"symbol name\" key is for symbols. The \"description\" key is for the description corresponding to the symbol. Please make sure the description is completed. The results should only have the dictionaries."},
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                      },
                    },
                  ],
                }
              ],
              max_tokens=4096,
            )
        return response
    except Exception as e:
        raise e
        
def process_gpt_response(response):
    extract_res = response.choices[0].message.content   
    try:
        gpt_json = str2json(extract_res)
        return gpt_json
    except (NameError, KeyError) as e:
        raise e

def gpt_extract_symbol_description(image_path, cur_attem):
    try:
        gpt_response = _call_gpt_api(image_path, cur_attem)
    except Exception as e:
        raise e

    try:
        sym_desc_pairs = process_gpt_response(gpt_response)
        return sym_desc_pairs
    except Exception as e:
        raise e
    

