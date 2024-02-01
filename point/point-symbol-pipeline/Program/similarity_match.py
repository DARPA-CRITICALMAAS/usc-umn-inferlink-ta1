from transformers import AutoModel, AutoConfig
import torch
from PIL import Image
from torchvision.transforms import functional as F
from transformers import AutoFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import glob
import os
import json

def vit_similarity(image_path1, image_path2):
    model_checkpoint = "google/vit-base-patch16-224-in21k"
    config = AutoConfig.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint, config=config)

    # Preprocess the images
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")

    # Resize and normalize the images
    image_size = (224, 224)
    image1 = F.resize(image1, image_size)
    image2 = F.resize(image2, image_size)
    image1 = F.to_tensor(image1).unsqueeze(0)
    image2 = F.to_tensor(image2).unsqueeze(0)

    # Extract features using the ViT model
    with torch.no_grad():
        features1 = model(image1).last_hidden_state.mean(dim=1)
        features2 = model(image2).last_hidden_state.mean(dim=1)

    # Calculate cosine similarity
    similarity_score = cosine_similarity(features1, features2)

    return similarity_score

def image_similarity(template_dir, legend_dir, rank = 1):

    template_directory = [file for file in os.listdir(template_dir) if file != ".DS_Store"]

    index_list_dict_rank = {}

    img_path_list = glob.glob(legend_dir + '/*.jpeg')
    img_path_list = sorted(img_path_list)

    for i, legend_image in enumerate(img_path_list):
        best_score = 0
        score_rank = {}

        legend_name = legend_image.split('/')[-1].split('.')[0]
        map_name = legend_name.split('_label_')[0]
        map_file_name = map_name + ".tif"
        pt_name_in_map = legend_name.split('_label_')[-1]

        for template_name in template_directory:
            template_image = os.path.join(template_dir, template_name)
            similarity_score = vit_similarity(legend_image, template_image)
            template_name = template_image.split('/')[-1].split('.')[0]
            score_rank[template_name] = similarity_score
            if similarity_score > best_score:
                best_score = similarity_score
                best_template_name = template_name
        # for best template
        # index_list_dict.setdefault(best_template_name, []).append(i)
        # print(legend_name + ' match with ' + best_template_name.split('&')[0] + ' with similarity score ' + str(best_score))

        # for top rank template
        top_template = sorted(score_rank, key=lambda k: score_rank[k], reverse=True)[:rank]
        print('\n'.join([f"Legend: {legend_name}, Template: {key}, Score: {value}" for key, value in sorted(score_rank.items(), key=lambda item: item[1], reverse=True)[:rank]]))
        print('*****************************************************************************************')
        for template_name in top_template:
            index_list_dict_rank.setdefault(template_name.split('&')[0], []).append(i)

    # print(index_list_dict)
    return index_list_dict_rank