"""
Script to perform Zero shot detection using Gemini Vision language models
Author: Bharath Santhanam
Email: bahrathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg


Description:
This script takes a test image and 5 nominal images as input and uses the prompt to generate an anomaly score for the test image
1 for anomaly and 0 for normal.
Usage. Currently this script is standalone and is not integrated with the main recovery pipeline.
Make sure to add Google API key in the script before running the script

Dependencies:
- google-generativeai library
- Google cloud account to get the API key

References:
This script is based on the github repo: google-gemini/generative-ai-python. Specific lines referred are mentioned in the code.
URL: https://github.com/google-gemini/generative-ai-python
Accessed on: 5/8/2024


"""

import PIL.Image
import google.generativeai as genai
import argparse

import ipdb
import json
import os
from arrange_img import generate_query_img
# LOok at the folder that has 5 nominal images and arrange them in the top row with size 100 x 100. Load the test image and arrange in the bottom row.
# make an image out of these two rows


def inference_gemini(prompt, img):
    """
    Function to perform inference using the Gemini Vision language model
    Args:
    prompt: str: Prompt for the zero shot detection
    img: PIL image: Image to be evaluated
    Returns:
    anomaly_score: int: Anomaly score for the test image 1 for anomaly and 0 for normaL
    """

    #line directly from 
    # https://github.com/google-gemini/generative-ai-python/blob/42d952a2429fa72c0f2617d71eb9aa3f708b6867/google/generativeai/__init__.py#L33
    model = genai.GenerativeModel("gemini-1.5-flash")
    # line directly from https://github.com/google-gemini/generative-ai-python/blob/42d952a2429fa72c0f2617d71eb9aa3f708b6867/google/generativeai/__init__.py#L34
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    # print(response.text)
    return int(response.text)


def main():
    parser = argparse.ArgumentParser(description="Extract features from the dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory that has many folders containing datasets",
        default="./test_images/",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        help="API key for the google cloud",
        default="XYZ",
    )

    parser.add_argument(
        "--save_features_dir",
        type=str,
        help="Directory to save the features",
        default="./save_test_images/",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    api_key = args.api_key
    #line directly from https://github.com/google-gemini/generative-ai-python/blob/42d952a2429fa72c0f2617d71eb9aa3f708b6867/google/generativeai/__init__.py#L31
    genai.configure(api_key=api_key)
    # read the json file and get the anomalous frames for the current folder

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        anomaly_scores_list = []
        for img in sorted(os.listdir(folder_path)):
            query_img_path = os.path.join(folder_path, img)
            image = generate_query_img(query_img_path)

            # Prompt for zero shot detection
            prompt = (
                "The images in the top row are examples of nominal door handle images,\
                    the image in the bottom row is a test image. Are there any anomalies in the test image?\
                    Normal conditions include: the door handle should be atleast partially visible, it may be in different lighting conditions, dark or distant\
                    and it may not be clearly visible due to blurriness.\
                    Answer 1 if there are anomalies:otherwise, answer 0."
            )

            anomaly_score = inference_gemini(prompt, image)
            print(anomaly_score)
            anomaly_scores_list.append(anomaly_score)


if __name__ == "__main__":
    main()
