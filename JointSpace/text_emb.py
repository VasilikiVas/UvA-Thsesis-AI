import json
import os

'''This script is used to extract the metadata of the listings from 
   the json files and store them in a list of dictionaries.'''
   
data = []
for file in os.listdir('/projects/0/vvasileiou4/ABO/listings/metadata'):
    if file.endswith('.json'):
        with open('/projects/0/vvasileiou4/ABO/listings/metadata/' + file) as f:
            for line in f:
                data.append(json.loads(line))

import csv

''' This script is used to create a list of 3d models that are available in the dataset.'''
list_3d_obj = []
with open("/projects/0/vvasileiou4/ABO/3dmodels/metadata/3dmodels.csv", 'r') as file:
  csvreader = csv.reader(file)
  flag = True
  for row in csvreader:
    if flag: 
      flag = False
    else:
      list_3d_obj.append(row[0])

''' This script is used to create a dictionary that maps the image ids to the image paths'''
image_path_dict = {}
with open("/projects/0/vvasileiou4/ABO/images/metadata/images.csv", 'r') as file:
  csvreader = csv.reader(file)
  flag = True
  for row in csvreader:
    if flag: 
      flag = False
    else:
      image_path_dict[row[0]] = row[3]

from googletrans import Translator
from googletrans.constants import LANGUAGES
import time

translator = Translator()

def get_color(i, color_languages):
    lang = data[i]['color'][0]['language_tag'].split('_')[0]
    val = data[i]['color'][0]['value']
    #print(lang, val)
    for t in color_languages:
        if t['language_tag'].split('_')[0] == 'en':
            print(t['value'])
            return t['value']
    print(translator.translate(val).text)
    return translator.translate(val).text

# Create dictionaries for each metadata field
color_dict = {}
material_dict = {}
label_dict = {}
fabric_dict = {}
image_id_dict = {}
other_img_id_dict = {}
a = []
# Iterate through the data and store the metadata in the dictionaries
for i in range(len(data)):
    # Check if the item is a 3d model
    if data[i]['item_id'] in list_3d_obj and data[i]['item_id'] not in a:
        # Check if the metadata field exists
        a.append(data[i]['item_id'])
        color_translation = False
        material_translation = False
        try:
            image_id_dict[data[i]['item_id']]= data[i]['main_image_id']
            label_dict[data[i]['item_id']]= data[i]['product_type'][0]['value']
            # Check if the color field has an english translation
            for c in data[i]['color']:
                if c['language_tag'].split('_')[0] == 'en':
                    color_dict[data[i]['item_id']]= c['value']
                    color_translation = True
                    break
            if not color_translation:
                color_dict[data[i]['item_id']]= translator.translate(data[i]['color'][0]['value']).text
            # Check if the material field has an english translation
            for m in data[i]['material']:
                if m['language_tag'].split('_')[0] == 'en':
                    material_dict[data[i]['item_id']]= m['value']
                    material_translation = True
                    break
            if not material_translation:
                material_dict[data[i]['item_id']]= translator.translate(data[i]['material'][0]['value']).text
            # Not used for now but can be used in the future
            fabric_dict[data[i]['item_id']]= data[i]['fabric_type'][0]['value']
            other_img_id_dict[data[i]['item_id']]= data[i]['other_image_id']
        except KeyError:
            pass

# Initialize an empty dictionary of dictionaries
intersect_dict = {}
# Loop through each key in the `label_dict` dictionary
for key in label_dict.keys():
    # Check if the key is present in the other dictionaries
    if key in color_dict.keys() and key in material_dict.keys() and key in image_id_dict.keys():
        # Create a list of other images directories
        other_img_list = []
        if key in other_img_id_dict.keys(): 
            for img in other_img_id_dict[key]:
                other_img_list.append(image_path_dict[img])
        # Create a new inner dictionary with the values from all three dictionaries
        inner_dict = {
            'image_id': image_path_dict[image_id_dict[key]],
            'label': label_dict[key],
            'color': color_dict[key],
            'material': material_dict[key],
            'other_img_id': other_img_list
        }
        # Add the inner dictionary to the intersection dictionary with the key from the `label_dict`
        intersect_dict[key] = inner_dict  
# Print the resulting intersection dictionary
print(len(intersect_dict))

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(input_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch_size, num_images, input_size)
        Returns:
            tensor of shape (batch_size, input_size) representing the
            attention-based combination of the input embeddings
        """
        batch_size, num_images, input_size = x.size()
        x = x.to(torch.float32)
        # compute attention weights
        x_flat = x.view(-1, input_size)
        attn = F.relu(self.attention_weights(x_flat)).view(batch_size, num_images, self.hidden_size)
        attn_weights = F.softmax(self.context_vector(attn), dim=1)
        # compute weighted sum of embeddings
        attn_weights = attn_weights.expand_as(x)
        weighted_emb = torch.sum(x * attn_weights, dim=1)
        return weighted_emb

from PIL import Image
import requests
from transformers import AutoTokenizer, CLIPTextModelWithProjection, AutoProcessor, CLIPVisionModelWithProjection
import torch

text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


for key, value in intersect_dict.items():
    label, color, material = value['label'], value['color'], value['material']

    text_inputs_des = tokenizer('a picture of ' + color+' '+material+' '+label, padding=True, return_tensors="pt")
    text_outputs_des = text_model(**text_inputs_des)
    text_embeds_des = text_outputs_des.text_embeds
    torch.save(text_embeds_des, '/projects/0/vvasileiou4/ABO/top_txt_emb/'+key+'_text.pt')