import json

from openai import OpenAI
import time
import re
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import requests
import base64

system_message_lane = """
You will act as the brain of an rule-based autonomous driving algorithm. You will be given the scenario description
about the motion of ego vehicle and surrounding vehicles in 20m. You should
- Choose a suitable decision from the given options.

Input:
- Your motion state
- Scenario description
- Decision Options
For example:
Your motion state: You current speed is 4.43m/s, acceleration is 0.24m/s^2, heading is 0.53pi.And your current speed limit is 6m/s.
Scenario description: You are driving in the middle lane of a multi-lanes road. The vehicle 08b6c23f26b15cf5 is driving in front_left of ego vehicle. The distance is 4.43. Velocity is 0.14m/s.
Decision options: keep the current lane, change lane to right

Output:
- Reasoning 
- Mode choice
For example: 
Reasoning: I should first analyse the surrounding objects. Then I will decide whether and which lane I should change.
--At First, I need observe the surrounding vehicles. There is only one vehicle in front left of me. The distance is 4.43m. 
Though my velocity is higher than it, I will not interact with it cause it is on my left. The safety has been satisfied.
--Secondly, There's no vehicle in directly front of me, so there's no need to change lane. I choose the mode keep. 
Decision choice: keep the current lane

Objects & Note:
- Avoid any collision with the other vehicles.
- Choose the Decision from the given options.
- Progress as further as you can, which means you can choose the fastest mode if you think it's ok, especially there are no vehicles around.
- if the velocity of the vehicle in directly front of you is much slower than yours and there are no vehicles on the lane you are changing to, you can change lane to overtake it.
"""

system_message_version='''
This is a Bird's Eye View map of a city road scene. The black lines represent the centerlines of the lanes, and all vehicles are driving along these centerlines. Different centerlines indicate different lanes. The area enclosed by blue lines is a pedestrian crosswalk. The square blocks colored #FF9E00 represent other vehicles, the ones colored #0000E6 represent pedestrians, and the square block colored #00CFBF represents the ego vehicle. The red arrows on each vehicle square represent the direction of vehicle movement. I want you to act as a professional driver.  Based on the image and the corresponding textual description, let's analyze the scene and provide driving recommendations for the ego vehicle. Please take a deep breath and work on this problem step-by-step. 
For example, you can first analyze the scene by answering the following question:
1. Which type is this scenario?
2. What is the status of the ego car? 
3. Is  there any car in the same lane of the ego car and in front of ego car?
Then, according to your analysis for this scenario, you need to give right instruction for ego-car planning by answering the following question:
Which is the right motion planning for ego car? "Change to the left lane, change to the right lane, or keep the current lane?"
Which is the right motion planning for ego car? "Accelerate, decelerate, or maintain the current speed?" 
'''

system_message = """
You will act as the brain of an rule-based autonomous driving algorithm. You will given a BEV image of current scenario and it's corresponding textual information surrounding the ego vehicle.
You are expected to accomplish two tasks: the first is to analyze the image and provide answers as required, while the second is to analyze the corresponding textual information provided and generate parameters for the planner. 
TASK ONE:
{
You will given a Bird's Eye View map of a city road scene. The black lines represent the centerlines of the lanes, and all vehicles are driving along these centerlines. Different centerlines indicate different lanes. The area enclosed by blue lines is a pedestrian crosswalk. The square blocks colored #FF9E00 represent other vehicles, the ones colored #0000E6 represent pedestrians, and the square block colored #00CFBF represents the ego vehicle. The red arrows on each vehicle square represent the direction of vehicle movement. I want you to act as a professional driver.  Based on the image and the corresponding textual description, let's analyze the scene and provide driving recommendations for the ego vehicle. Please take a deep breath and work on this problem step-by-step. 
For example, you can first analyze the scene by answering the following question:
1. Which type is this scenario?
2. What is the status of the ego car? 
3. Is  there any car in the same lane of the ego car (Vehicles are crossed by the same black line) and in front of ego car?
Then, according to your analysis for this scenario, you need to give right instruction for ego-car planning by answering the following question:
Which is the right motion planning for ego car? "Change to the left lane, change to the right lane, or keep the current lane?"
Which is the right motion planning for ego car? "Accelerate, decelerate, or maintain the current speed?" 

Input: the encode of a image in user message
Output: the answer of TASK ONE

Note:

Sometimes, arrows might overlap with other objects, so it's crucial to pay special attention to the red marks when determining the direction of vehicle movement.
You should Keep your output not exceed 100 words.
}

TASK TWO:
{
Base on the analysis of TASK ONE, you will be given the scenario description
about the motion of ego vehicle and surrounding vehicles in 30m. And you will be given several decision options making 
in the longitudinal and lateral direction. You should
- Choose the appropriate speed limit(m/s) max acceleration(m/s^2), and max deceleration(m/s^2) that can satisfy the objects.
- Choose the lateral option that can satisfy the objects from given options.

Input:
- Your motion state
- Scenario description
For example:
Your motion state: You current speed is 4.43m/s, acceleration is 0.24m/s^2, heading is 0.53pi.And your current speed limit is 6m/s.
Scenario description: You are driving in the middle lane of a multi-lanes road. The vehicle 08b6c23f26b15cf5 is driving in front_left of ego vehicle. The distance is 4.43. Velocity is 0.14m/s.
Speed limit options: [3.0,9.0,15.0]
Max acceleration options: [1.5,2.5,3.5]
Max deceleration options: [1.0,2.0,3.0] default is 3.0
Lateral options: keep the current lane, change lane to right

Output:
- Reasoning
- Speed limit, max acceleration and decision choice
For example: 
Reasoning: I should first analyse the surrounding objects. Then I will decide whether I should change lane and choose a suitable speed mode,and then other parameters.
--At First, I need observe the surrounding vehicles. There is only one vehicle in front left of me. The distance is 4.43m. 
Though my velocity is higher than it, I will not interact with it cause it is on my left. The safety has been satisfied.
--Secondly, There's no vehicle in directly front of me, so there's no need to change lane. I choose the mode keep. 
--Finally, I will make the ego vehicle progress as far as possible. My current speed is 4.43m/s, and I will not interact with other vehicles
in the future 2s. So I can choose a faster Speed limit choice, which is 15.0. 
To arrive a high speed, I should further accelerate and choose a max acceleration near the current value. So I choose 1.5.
Since my current status is acceleration, I choose default max deceleration as 3.0.
Lateral choice: keep the current lane
Speed limit choice: 15.0
Max acceleration choice: 1.5
Max deceleration choice: 3.0


Note:
- Avoid any collision with the other vehicles.
- You must choose the lateral option, speed limit choice max acceleration and max deceleration from the given options.
- Keep the format as same as the example.
- Progress as further as you can, which means you can choose the fastest speed limit if you think it's ok, especially there are no vehicles around.
- If the velocity of the vehicle in directly front of you is much slower than yours and there are no vehicles on the lane you are changing to, you can change lane to overtake it. 
- But if your front vehicle is waiting or just was waiting for red light, you had better keep the current lane.
- Speed limit is not the target you should reach, but the target you cann't exceed.
- Max acceleration and max deceleration are the targets you want to reach in 1s. To ensure comfort, you should choose a value near the current values.
- If you are accelerating, you can choose the default max deceleration. else if you are decelerating, you should choose a value near the current value.

You should Keep your output not exceed 150 words, and place your four choices at the end of the answer 
}
"""

# system_message='''
# You will act as the brain of an rule-based autonomous driving algorithm. You will be given the scenario description
# about the motion of ego vehicle and surrounding vehicles in 20m. And you will be given several decision options making 
# in the longitudinal and lateral direction. You should
# - Choose the appropriate speed limit(m/s) max acceleration(m/s^2), and max deceleration(m/s^2) that can satisfy the objects.
# - Choose the lateral option that can satisfy the objects from given options.

# Input:
# - Your motion state
# - Scenario description
# For example:
# Your motion state: You current speed is 4.43m/s, acceleration is 0.24m/s^2, heading is 0.53pi.And your current speed limit is 6m/s.
# Scenario description: You are driving in the middle lane of a multi-lanes road. The vehicle 08b6c23f26b15cf5 is driving in front_left of ego vehicle. The distance is 4.43. Velocity is 0.14m/s.
# Speed limit options: [3.0,9.0,15.0]
# Max acceleration options: [1.0,1.5,2.0]
# Max deceleration options: [1.0,2.0,3.0] default is 3.0
# Lateral options: keep the current lane, change lane to right

# Output:
# - Reasoning
# - Speed limit, max acceleration and decision choice
# For example: 
# Reasoning: I should first analyse the surrounding objects. Then I will decide whether I should change lane and choose a suitable speed mode,and then other parameters.
# --At First, I need observe the surrounding vehicles. There is only one vehicle in front left of me. The distance is 4.43m. 
# Though my velocity is higher than it, I will not interact with it cause it is on my left. The safety has been satisfied.
# --Secondly, There's no vehicle in directly front of me, so there's no need to change lane. I choose the mode keep. 
# --Finally, I will make the ego vehicle progress as far as possible. My current speed is 4.43m/s, and I will not interact with other vehicles
# in the future 2s. So I can choose a faster Speed limit choice, which is 15.0. 
# To arrive a high speed, I should further accelerate and choose a max acceleration near the current value. So I choose 1.5.
# Since my current status is acceleration, I choose default max deceleration as 3.0.
# Lateral choice: keep the current lane
# Speed limit choice: 15.0
# Max acceleration choice: 1.5
# Max deceleration choice: 3.0


# Objects & Note:
# - Avoid any collision with the other vehicles.
# - You must choose the lateral option, speed limit choice max acceleration and max deceleration from the given options.
# - Keep the format as same as the example
# - - If the velocity of the vehicle in directly front of you is much slower than yours and there are no vehicles on the lane you are changing to, you can change lane to overtake it.
# - Speed limit is not the target you should reach, but the target you cann't exceed.
# - Max acceleration and max deceleration are the targets you want to reach in 1s. To ensure comfort, you should choose a value near the current values.
# - If you are accelerating, you can choose the default max deceleration. else if you are decelerating, you should choose a value near the current value.
# - Progress as further as you can, which means you can choose the fastest speed limit if you think it's ok, especially there are no vehicles around.
# '''

apitoken = ""
url = ''

# @retry(wait=wait_random_exponential(min=60, max=120), stop=stop_after_attempt(6))
# def completion_with_backoff(**kwargs):
#     return openai.ChatCompletion.create(**kwargs)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_gpt_decision(user_message, iteration, image_path,try_times=3):
    client = OpenAI(base_url=url, api_key=apitoken)
    lane_mode = "keep"
    speed_mode = -1
    accel_mode = -1
    decel_mode = -1
    sys_message = system_message
    while try_times:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": sys_message},
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type":"image","image":encode_image(image_path)},
                        {"type": "text", "text": user_message},
                    ]
                }
            ]
        )
        # print(response)
        result=response.choices[0].message.content

        result_casefold = result.casefold()
        answer_lane_s = result_casefold.rfind("lateral".casefold())
        answer_speed_s = result_casefold.rfind("speed".casefold())
        answer_accel_s = result_casefold.rfind("acceleration".casefold())
        answer_decel_s = result_casefold.rfind("deceleration".casefold())

        if not answer_lane_s == -1:
            answer_lane = result[answer_lane_s:answer_speed_s]
            if "left" in answer_lane:
                lane_mode = "left"
            elif "right" in answer_lane:
                lane_mode = "right"
            else:
                lane_mode = "keep"
        if not answer_speed_s == -1:
            answer_speed = result[answer_speed_s:answer_accel_s]
            if "3.0" in answer_speed:
                speed_mode = 0
            elif "9.0" in answer_speed:
                speed_mode = 1
            elif "15.0" in answer_speed:
                speed_mode = 2
        if not answer_accel_s == -1:
            answer_accel = result[answer_accel_s:answer_decel_s]
            if "1.5" in answer_accel:
                accel_mode = 0
            elif "2.5" in answer_accel:
                accel_mode = 1
            elif "3.5" in answer_accel:
                accel_mode = 2
        if not answer_decel_s == -1:
            answer_decel = result[answer_decel_s:]
            if "1.0" in answer_decel:
                decel_mode = 0
            elif "2.0" in answer_decel:
                decel_mode = 1
            elif "3.0" in answer_decel:
                decel_mode = 2
        if not speed_mode == -1 and not accel_mode == -1 and not decel_mode == -1:
            break
        try_times -= 1

    return lane_mode, speed_mode, accel_mode, decel_mode
