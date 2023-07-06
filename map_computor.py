# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz5038

1) interacting with SUMO, including
      retrive values, set lights
2) interacting with sumo_agent, including
      returning status, rewards, etc.

'''

import numpy as np
import math
import os
import sys
import pandas as pd 
    
import xml.etree.ElementTree as ET
from sys import platform
from sumo_agent import Vehicles
from datetime import datetime
import statistics

###### Please Specify the location of your traci module


yeta = 0.15
tao = 2
constantC = 40.0
# carWidth = 3.3
grid_width = 4
area_length = 600


lenpahse = 172
phases_light_7 = ['RG_SG_SG_SG_LG_SG_LG_RG_LG_TG_RG_SG_RG_SG_SG_LG_SG_LG_RG_LG_TG_RG', 'SG_SG_LG_SG_LG_RG_LG_TG_RG_SG_SG_SG_SG_LG_RG_LG_TG_RG', 'SG_SG_LG_SG_LG_RG_LG_RG_TG_RG_SG_LG_SG_LG_LG_RG_TG', 'RG_RG_RG_SG_LG_SG_LG_TG_RG_LG_RG_LG_RG_RG_SG_SG_SG_TG_SG', 'LG_RG_LG_RG_SG_RG_SG_SG_LG_SG_LG_TG_RG_RG_LG_SG_RG_SG_RG_SG_SG_LG_TG', 'SG_RG_SG_SG_LG_SG_LG_RG_LG_RG_TG_LG_SG_RG_SG_SG_LG_LG_LG_SG_TG_SG', 'LG_RG_RG_RG_SG_SG_SG_LG_LG_TG_RG_LG_RG_SG_RG_SG_SG_SG_RG_LG_TG_RG', 'RG_RG_RG_SG_SG_LG_LG_TG_RG_RG_SG_SG_SG_RG_LG_TG_RG', 'RG_SG_SG_LG_SG_LG_RG_LG_RG_TG_RG_SG']
numphases = len(phases_light_7)
node_light_7 = 'cluster_11791147_1259892844_1259892853_1259892859_2032404487_2535135910_27789090_320247058_7123869611_7123869612_81289422_81290972_996182408'
listLanes = {'177513122#0_1', '177513122#0_2', '317245904#0_2', '762416404#1_0', '-806118899#2_0', '110282956#2_0', '-806118902#2_0', '317209212#0_3', '-1052446122_0', '806118898#2_0', '-806118899#0_0', '127375467#1_2', '317245904#0_3', '619973576#1_4', '317209212#0_0', '246476583#0_0', '127375467#1_0', '127375383#1_0', '806118903_0', '317209212#0_2', '619973576#1_1', '127375467#1_1', '317209212#0_1', '246476583#1_0', '893990362#1_0', '893990362#1_1', '-246476583#0_0', '177513122#0_0', '806118899#2_0', '-806118898#0_0', '127375383#1_1', '894002830#1_1', '619973576#1_2', '619973576#1_0', '-246476583#1_0', '894002830#1_0', '255736353_0', '806118902#2_0', '-1052446121_0', '317245904#0_1', '619973576#1_3', '317245904#0_0', '-806118902#0_0'}

'''
input: phase "NSG_SNG" , four lane number, in the key of W,E,S,N
output: 
1.affected lane number: 4_0_0, 4_0_1, 3_0_0, 3_0_1
# 2.destination lane number, 0_3_0,0_3_1  

'''
current_time = 0 
def get_current_time():
    global current_time 
    #print (current_time)
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_current_time.csv')
    time = df.iloc[current_time,0]  
    current_time += 1
    return time 

# def phase_affected_lane(phase="NSG_SNG",
#                         four_lane_ids={'W': 'edge1-0', "E": "edge2-0", 'S': 'edge4-0', 'N': 'edge3-0'}):
#     directions = phase.split('_')
#     affected_lanes = []
#     for direction in directions:
#         for k, v in four_lane_ids.items():
#             if v.strip() != '' and direction.startswith(k):
#                 for lane_no in direction_lane_dict[direction]:
#                     affected_lanes.append("%s_%d" % (v, lane_no))
#                     # affacted_lanes.append("%s_%d" % (v, 0))
#     if affected_lanes == []:
#         raise("Please check your phase and lane_number_dict in phase_affacted_lane()!")
#     return affected_lanes


'''
input: central nodeid "node0", surrounding nodes WESN: [1,2,3,4]
output: four_lane_ids={'W':'edge1-0',"E":"edge2-0",'S':'edge4-0','N':'edge3-0'})
'''


# def find_surrounding_lane_WESN(central_node_id=node_light_7, WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
#     tree = ET.parse('./data/one_run/m/tallinn.net.xml')
#     root = tree.getroot()
#     four_lane_ids_dict = {}
#     for k, v in WESN_node_ids.items():
#         four_lane_ids_dict[k] = root.find("./edge[@from='%s'][@to='%s']" % (v, central_node_id)).get('id')
#     return four_lane_ids_dict


'''
coordinate mapper
'''


def coordinate_mapper(x1, y1, x2, y2, area_length=600, area_width=600):
    x1 = int(x1 / grid_width)
    y1 = int(y1 / grid_width)
    x2 = int(x2 / grid_width)
    y2 = int(y2 / grid_width)
    x_max = x1 if x1 > x2 else x2
    x_min = x1 if x1 < x2 else x2
    y_max = y1 if y1 > y2 else y2
    y_min = y1 if y1 < x2 else y2
    length_num_grids = int(area_length / grid_width)
    width_num_grids = int(area_width / grid_width)
    return length_num_grids - y_max, length_num_grids - y_min, x_min, x_max

# def get_phase_affected_lane_traffic_max_volume(phase="NSG_SNG", tl_node_id=node_light_7,
#                                  WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
#     four_lane_ids_dict = find_surrounding_lane_WESN(central_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
#     directions = phase.split('_')
#     traffic_volume_start_end = []
#     for direction in directions:
#         traffic_volume_start_end.append([four_lane_ids_dict[direction[0]],four_lane_ids_dict[direction[1]]])
#     tree = ET.parse('./data/one_run/m/osm_pt.rou.xml')
#     root = tree.getroot()
#     phase_volumes = []
#     for lane_id in traffic_volume_start_end:
#         to_lane_id="edge%s-%s"%(lane_id[1].split('-')[1],lane_id[1].split('-')[0][4:])
#         time_begin = root.find("./flow[@from='%s'][@to='%s']" % (lane_id[0], to_lane_id)).get('begin')
#         time_end = root.find("./flow[@from='%s'][@to='%s']" % (lane_id[0], to_lane_id)).get('end')
#         volume = root.find("./flow[@from='%s'][@to='%s']" % (lane_id[0], to_lane_id)).get('number')
#         phase_volumes.append((float(time_end)-float(time_begin))/float(volume))
#     return max(phase_volumes)


# def phase_affected_lane_position(phase="NSG_SNG", tl_node_id=node_light_7,
#                                  WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
#     '''
#     input: NSG_SNG ,central nodeid "node0", surrounding nodes WESN: {"W":"1", "E":"2", "S":"3", "N":"4"}
#     output: edge-ids, 4_0_0, 4_0_1, 3_0_0, 3_0_1
#     [[ 98,  100,  204,  301],[ 102, 104, 104, 198]]
#     '''
#     four_lane_ids_dict = find_surrounding_lane_WESN(central_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
#     affected_lanes = phase_affected_lane(phase=phase, four_lane_ids=four_lane_ids_dict)
#     tree = ET.parse('./data/cross.net.xml')
#     root = tree.getroot()
#     indexes = []
#     for lane_id in affected_lanes:
#         lane_shape = root.find("./edge[@to='%s']/lane[@id='%s']" % (tl_node_id, lane_id)).get('shape')
#         lane_x1 = float(lane_shape.split(" ")[0].split(",")[0])
#         lane_y1 = float(lane_shape.split(" ")[0].split(",")[1])
#         lane_x2 = float(lane_shape.split(" ")[1].split(",")[0])
#         lane_y2 = float(lane_shape.split(" ")[1].split(",")[1])
#         ind_x1, ind_x2, ind_y1, ind_y2 = coordinate_mapper(lane_x1, lane_y1, lane_x2, lane_y2)
#         indexes.append([ind_x1, ind_x2 + 1, ind_y1, ind_y2 + 1])
#     return indexes


# def phases_affected_lane_postions(phases=["NSG_SNG_NWG_SEG", "NEG_SWG_NWG_SEG"], tl_node_id=node_light_7,
#                                   WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
#     parameterArray = []
#     for phase in phases:
#         parameterArray += phase_affected_lane_position(phase=phase, tl_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
#     return parameterArray


def vehicle_location_mapper(coordinate, area_length=600, area_width=600):
    transformX = math.floor(coordinate[0] / grid_width)
    transformY = math.floor((area_length - coordinate[1]) / grid_width)
    length_num_grids = int(area_length/grid_width)
    transformY = length_num_grids-1 if transformY == length_num_grids else transformY
    transformX = length_num_grids-1 if transformX == length_num_grids else transformX
    tempTransformTuple = (transformY, transformX)
    return tempTransformTuple


def translateAction(action):
    result = 0
    for i in range(len(action)):
        result += (i + 1) * action[i]
    return result


ind_next_phase = 0 
ind_next_phase_time_eclipsed = 0
def changeTrafficLight_7(current_phase=0):  # [WNG_ESG_WSG_ENG_NWG_SEG]
    global ind_next_phase 
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/changeTrafficLight_7_next_phase.csv')
    next_phase = df.iloc[ind_next_phase,0]  
    ind_next_phase += 1

    global ind_next_phase_time_eclipsed 
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/changeTrafficLight_7_next_phase_time_eclipsed.csv')
    next_phase_time_eclipsed = df.iloc[ind_next_phase_time_eclipsed,0]  
    ind_next_phase_time_eclipsed += 1

    return next_phase, next_phase_time_eclipsed








def calculate_reward(tempLastVehicleStateList):
    waitedTime = 0
    stop_count = 0
    for key, vehicle_dict in tempLastVehicleStateList.items():
        if tempLastVehicleStateList[key]['speed'] < 5:
            waitedTime += 1
            #waitedTime += (1 +math.pow(tempLastVehicleStateList[key]['waitedTime']/50,2))
        if tempLastVehicleStateList[key]['former_speed'] > 0.5 and tempLastVehicleStateList[key]['speed'] < 0.5:
            stop_count += (tempLastVehicleStateList[key]['stop_count']-tempLastVehicleStateList[key]['former_stop_count'])
    #PI = (waitedTime + 10 * stop_count) / len(tempLastVehicleStateList) if len(tempLastVehicleStateList)!=0 else 0
    PI = waitedTime/len(tempLastVehicleStateList) if len(tempLastVehicleStateList)!=0 else 0
    return - PI

ind_mapsOfCars = 0
def getMapOfVehicles(area_length=600):
    global ind_mapsOfCars
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/getMapOfVehicles'+str(ind_mapsOfCars)+'.csv')
    mapOfCars=df.values
    return mapOfCars

def restrict_reward(reward,func="unstrict"):
    if func == "linear":
        bound = -50
        reward = 0 if reward < bound else (reward/(-bound) + 1)
    elif func == "neg_log":
        reward = math.log(abs(reward)+1)
    else:
        pass
    
    return reward


def log_rewards(vehicle_dict, action, rewards_info_dict, file_name, timestamp,rewards_detail_dict_list):

    reward, reward_detail_dict = get_rewards_from_sumo(vehicle_dict, action, rewards_info_dict)
    list_reward_keys = np.sort(list(reward_detail_dict.keys()))
    reward_str = "{0}, {1}".format(timestamp,action)
    for reward_key in list_reward_keys:
        reward_str = reward_str + ", {0}".format(reward_detail_dict[reward_key][2])
    reward_str += '\n'

    fp = open(file_name, "a")
    fp.write(reward_str)
    fp.close()
    rewards_detail_dict_list.append(reward_detail_dict)

import ast
ind_reward = 0
ind_reward_detail_dict = 0
def get_rewards_from_sumo(vehicle_dict, action, rewards_info_dict,
                          listLanes=listLanes):
    global ind_get_vehicle_id_entering
    ind_get_vehicle_id_entering +=1
    global ind_overall_queue_length
    ind_overall_queue_length +=1
    global ind_overall_waiting_time
    ind_overall_waiting_time += 1
    global ind_overall_delay
    ind_overall_delay += 1
    global ind_get_num_of_emergency_stops
    ind_get_num_of_emergency_stops += 1
    global ind_get_travel_time_duration
    ind_get_travel_time_duration += 1
    global ind_get_partial_travel_time_duration
    ind_get_partial_travel_time_duration += 1
    global ind_get_vehicle_id_leaving
    ind_get_vehicle_id_leaving += 1
    ind_get_travel_time_duration += 1

    global ind_reward
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_rewards_from_sumo_reward.csv')
    reward = df.iloc[ind_reward,0].tolist() 
    ind_reward += 1

    global ind_reward_detail_dict 
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_rewards_from_sumo_dic_reward.csv')
    reward_detail_dict = ast.literal_eval(df.iloc[ind_reward_detail_dict,1]) 
    ind_reward_detail_dict += 1

    return reward, reward_detail_dict

def get_rewards_from_dict_list(rewards_detail_dict_list):
    reward = 0
    for i in range(len(rewards_detail_dict_list)):
        for k, v in rewards_detail_dict_list[i].items():
            if v[0]:  # True or False
                reward += v[1] * v[2]
    reward = restrict_reward(reward)
    return reward

ind_overall_queue_length=0
def get_overall_queue_length(listLanes):
    global ind_overall_queue_length
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/Sumo_record_data/one_run/m/get_overall_queue_length.csv')
    overall_queue_length = df.iloc[ind_overall_queue_length,0]  
    ind_overall_queue_length += 1
    return overall_queue_length

ind_overall_noise=0
def get_overall_noise(listLanes):
    global ind_overall_noise
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_overall_noise.csv')
    overall_noise = df.iloc[ind_overall_noise,0]  
    ind_overall_noise += 1
    return overall_noise

ind_overall_waiting_time=0
def get_overall_waiting_time(listLanes):
    global ind_overall_waiting_time
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_overall_waiting_time.csv')
    overall_waiting_time = df.iloc[ind_overall_waiting_time,0]  
    ind_overall_waiting_time += 1
    return overall_waiting_time

ind_overall_delay=0
def get_overall_delay(listLanes):
    global ind_overall_delay
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_overall_delay.csv')
    overall_delay = df.iloc[ind_overall_delay,0]  
    ind_overall_delay += 1
    return overall_delay

def get_flickering(action):
    return action

# calculate number of emergency stops by vehicle
ind_get_num_of_emergency_stops=0
def get_num_of_emergency_stops(vehicle_dict):
    global ind_get_num_of_emergency_stops
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_num_of_emergency_stops.csv')
    get_num_of_emergency_stops = df.iloc[ind_get_num_of_emergency_stops,0]  
    ind_get_num_of_emergency_stops += 1
    return get_num_of_emergency_stops

ind_get_partial_travel_time_duration=0
def get_partial_travel_time_duration(vehicle_dict, vehicle_id_list):
    global ind_get_num_of_emergency_stops
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_partial_travel_time_duration.csv')
    get_partial_travel_time_duration = df.iloc[ind_get_partial_travel_time_duration,0]  
    ind_get_partial_travel_time_duration += 1
    return get_partial_travel_time_duration


ind_get_travel_time_duration=0
def get_travel_time_duration(vehicle_dict, vehicle_id_list):
    global ind_get_travel_time_duration
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_travel_time_duration.csv')
    get_travel_time_duration = df.iloc[ind_get_travel_time_duration,0]  
    ind_get_travel_time_duration += 1
    return get_travel_time_duration

ind_dic_vehicles=0
def update_vehicles_state(dic_vehicles):
    global ind_get_vehicle_id_entering
    ind_get_vehicle_id_entering += 1
    
    global ind_dic_vehicles 
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/update_vehicles_state.csv')
    dic_vehicles = df.iloc[ind_dic_vehicles,0]
    ind_dic_vehicles += 1
    
    return dic_vehicles


ind_laneQueueTracker = 0
ind_laneNumVehiclesTracker=0
ind_laneWaitingTracker = 0
ind_laneNoiseTracker = 0
ind_co2Tracker = 0
ind_fuelTracker = 0 
ind_waiting =0
ind_laneNoisePredictTracker = 0
ind_laneSpeedTracker = 0
ind_laneTravelTimeTracker = 0
ind_len_listLanes =0

def status_calculator_noise(noisemodel):
    global ind_mapsOfCars
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/getMapOfVehicles'+str(ind_mapsOfCars)+'.csv')
    mapOfCars=df.values
    ind_mapsOfCars += 1

    global ind_laneQueueTracker
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/laneQueueTracker.csv')
    laneQueueTracker = df.iloc[43*ind_laneQueueTracker:43*(ind_laneQueueTracker+1),1].tolist()
    ind_laneQueueTracker += 1

    global ind_laneNumVehiclesTracker
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/laneNumVehiclesTracker.csv')
    laneNumVehiclesTracker = df.iloc[43*ind_laneNumVehiclesTracker:43*(ind_laneNumVehiclesTracker+1),1].tolist()
    ind_laneNumVehiclesTracker += 1

    global ind_laneWaitingTracker
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/laneWaitingTracker.csv')
    laneWaitingTracker = df.iloc[43*ind_laneWaitingTracker:43*(ind_laneWaitingTracker+1),1].tolist()
    ind_laneWaitingTracker += 1

    global ind_laneNoiseTracker
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/laneNoiseTracker.csv')
    laneNoiseTracker = df.iloc[43*ind_laneNoiseTracker:43*(ind_laneNoiseTracker+1),1].tolist()
    ind_laneNoiseTracker += 1

    global ind_co2Tracker
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/co2Tracker.csv')
    co2Tracker = df.iloc[43*ind_co2Tracker:43*(ind_co2Tracker+1),1].tolist()
    ind_co2Tracker += 1

    global ind_fuelTracker
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/fuelTracker.csv')
    fuelTracker = df.iloc[43*ind_fuelTracker:43*(ind_fuelTracker+1),1].tolist()
    ind_fuelTracker += 1

    global ind_waiting
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/fuelTracker.csv')
    waiting = df.iloc[43*ind_waiting:43*(ind_waiting+1),1].tolist()
    ind_waiting += 1

    global ind_laneNoisePredictTracker
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/laneNoisePredictTracker.csv')
    laneNoisePredictTracker = df.iloc[43*ind_laneNoisePredictTracker:43*(ind_laneNoisePredictTracker+1),1].tolist()
    ind_laneNoisePredictTracker += 1

    global ind_laneSpeedTracker
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/laneSpeedTracker.csv')
    laneSpeedTracker = df.iloc[43*ind_laneSpeedTracker:43*(ind_laneSpeedTracker+1),1].tolist()
    ind_laneSpeedTracker += 1

    global ind_laneTravelTimeTracker
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/laneTravelTimeTracker.csv')
    laneTravelTimeTracker = df.iloc[43*ind_laneTravelTimeTracker:43*(ind_laneTravelTimeTracker+1),1].tolist()
    ind_laneTravelTimeTracker += 1

    global ind_len_listLanes 
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/status_calculator_noise/len_listLanes.csv')
    len_listLanes = df.iloc[ind_len_listLanes,0]  
    ind_len_listLanes += 1

    return (laneQueueTracker,laneNumVehiclesTracker,laneWaitingTracker,mapOfCars,laneNoiseTracker,co2Tracker,fuelTracker,waiting,laneNoisePredictTracker,laneSpeedTracker,laneTravelTimeTracker,len_listLanes)


ind_get_vehicle_id_entering=0
def get_vehicle_id_entering():
    global ind_get_vehicle_id_entering
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_vehicle_id_entering.csv')
    vehicle_id_entering = df.iloc[:,ind_get_vehicle_id_entering].tolist()
    ind_get_vehicle_id_entering += 1
    return vehicle_id_entering

ind_get_vehicle_id_leaving =0
def get_vehicle_id_leaving(vehicle_dict):
    global ind_get_vehicle_id_leaving
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_vehicle_id_leaving.csv')
    vehicle_id_leaving = df.iloc[:,ind_get_vehicle_id_leaving].tolist()
    ind_get_vehicle_id_leaving += 1
    return vehicle_id_leaving



# def get_car_on_red_and_green(cur_phase):
#     listLanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2',
#                  'edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
#     vehicle_red = []
#     vehicle_green = []
#     if cur_phase == 1:
#         red_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']
#         green_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
#     else:
#         red_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
#         green_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']
#     for lane in red_lanes:
#         vehicle_red.append(traci.lane.getLastStepVehicleNumber(lane))
#     for lane in green_lanes:
#         vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
#         omega = 0
#         for vehicle_id in vehicle_ids:
#             traci.vehicle.subscribe(vehicle_id, (tc.VAR_DISTANCE, tc.VAR_LANEPOSITION))
#             distance = traci.vehicle.getSubscriptionResults(vehicle_id).get(132)
#             if distance > 100:
#                 omega += 1
#         vehicle_green.append(omega)

#     return max(vehicle_red), max(vehicle_green)

def get_status_img(current_phase,tl_node_id=node_light_7,area_length=600):
    mapOfCars = getMapOfVehicles(area_length=area_length)

    current_observation = [mapOfCars]
    return current_observation

def set_yellow(dic_vehicles,rewards_info_dict,f_log_rewards,rewards_detail_dict_list,node_id=node_light_7):
    Yellow = ''
    for i in range(lenpahse):
        Yellow = Yellow+"y"
    for i in range(3):
        global current_time 
        df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_current_time.csv')
        timestamp = df.iloc[current_time,0]  
        log_rewards(dic_vehicles, 0, rewards_info_dict, f_log_rewards, timestamp, rewards_detail_dict_list)
        update_vehicles_state(dic_vehicles)

def set_all_red(dic_vehicles,rewards_info_dict,f_log_rewards,rewards_detail_dict_list,node_id=node_light_7):
    Red = ''
    for i in range(lenpahse):
        Red = Red+"r"
    for i in range(3):
        global current_time 
        df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_current_time.csv')
        timestamp = df.iloc[current_time,0]  
        log_rewards(dic_vehicles, 0, rewards_info_dict, f_log_rewards, timestamp,rewards_detail_dict_list)
        update_vehicles_state(dic_vehicles)

def run(action, current_phase, current_phase_duration, vehicle_dict, rewards_info_dict, f_log_rewards, rewards_detail_dict_list,node_id=node_light_7):

    return_phase = current_phase
    return_phase_duration = current_phase_duration

    if action == 1:
        set_yellow(vehicle_dict,rewards_info_dict,f_log_rewards, rewards_detail_dict_list,node_id=node_id)
        # set_all_red(vehicle_dict,rewards_info_dict,f_log_rewards, node_id=node_id)
        return_phase, _ = changeTrafficLight_7(current_phase=current_phase)  # change traffic light in SUMO according to actionToPerform
        return_phase_duration = 0
        
    global current_time 
    df = pd.read_csv('Sumo_record_data/records_sumo/one_run/m/get_current_time.csv')
    timestamp = df.iloc[current_time,0]  
    log_rewards(vehicle_dict, action, rewards_info_dict, f_log_rewards, timestamp, rewards_detail_dict_list)
    vehicle_dict = update_vehicles_state(vehicle_dict)
    return return_phase, return_phase_duration+1, vehicle_dict



# def get_base_min_time(traffic_volumes,min_phase_time):
#     traffic_volumes=np.array([36,72,0])
#     min_phase_times=np.array([10,35,35])
#     for i, min_phase_time in enumerate(min_phase_times):
#         ratio=min_phase_time/traffic_volumes[i]
#         traffic_volumes_ratio=traffic_volumes/ratio

# def phase_vector_to_number(phase_vector,phases_light=phases_light_7):
#     phase_vector_7 = []
#     result = -1
#     for i in range(len(phases_light)):
#         phase_vector_7.append(str(get_phase_vector(i)))
#     if phase_vector in phase_vector_7:
#         return phase_vector_7.index(phase_vector)
#     else:
#         raise ("Phase vector %s is not in phases_light %s"%(phase_vector,str(phase_vector_7)))



if __name__ == '__main__':
    pass
#     print(get_phase_vector(0))
#     print(get_phase_vector(1))
 
#     pass
    # traci.close()
