import socket
import requests
import math
import time
from geopy.geocoders import Nominatim
import cv2
import threading
import numpy as np
import subprocess
import os
from dotenv import load_dotenv
from collections import deque


PHONE_IP = "10.202.51.16"
PHONE_PORT = 8080

load_dotenv()

def speak(text):
    os.system('espeak "{}"'.format(text))


def meters_to_steps(meters):
    return round(meters / 0.7)

def format_steps(steps):
    if steps < 5:
        return "a few steps"
    elif steps < 10:
        return "about ten steps"
    elif steps < 20:
        return f"about {round(steps/5)*5} steps"  
    elif steps < 100:
        return f"about {round(steps/10)*10} steps"  
    else:
        return f"about {round(steps/50)*50} steps" 

def parse_nmea_sentence(sentence):
    if sentence.startswith("$GPGGA"):
        parts = sentence.split(",")
        if len(parts) > 5:
            try:
                lat = float(parts[2][:2]) + float(parts[2][2:]) / 60
                if parts[3] == "S":
                    lat = -lat
                lon = float(parts[4][:3]) + float(parts[4][3:]) / 60
                if parts[5] == "W":
                    lon = -lon
                return lat, lon
            except (ValueError, IndexError):
                return None, None
    return None, None

def get_gps_location():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect((PHONE_IP, PHONE_PORT))
            print("Connected to GPS stream...")
            
            attempts = 0
            while attempts < 5:
                data = s.recv(1024).decode('utf-8').strip()
                if not data:
                    break
                
                for line in data.splitlines():
                    lat, lon = parse_nmea_sentence(line)
                    if lat and lon:
                        speak("GPS signal acquired")
                        return lat, lon
                
                attempts += 1
                time.sleep(1)
            
            print("GPS signal not found. Using approximate location.")
            return None
    except Exception as e:
        print("GPS connection failed")
        return None

def get_ip_location():
    try:
        response = requests.get("http://ipinfo.io/json")
        data = response.json()
        location = data['loc'].split(',')
        return float(location[0]), float(location[1])
    except Exception as e:
        return None

def get_current_location():
    gps_location = get_gps_location()
    if gps_location:
        return gps_location
    
    print("Using approximate location")
    ip_location = get_ip_location()
    if ip_location:
        return ip_location
    
    print("Unable to determine location")
    return None

def get_route_instructions(start_lat, start_lon, end_lat, end_lon):
    API_KEY = os.getenv("API_KEY")

    url = "https://api.openrouteservice.org/v2/directions/foot-walking"
    
    headers = {
        "Authorization": API_KEY,
        "Accept": "application/json, application/geo+json"
    }
    
    params = {
        "start": f"{start_lon},{start_lat}",
        "end": f"{end_lon},{end_lat}",
        "instructions": "true",
        "language": "en"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        route_data = response.json()
        
        if 'features' not in route_data or not route_data['features']:
            print("No route found")
            return None, None
        
        steps = route_data['features'][0]['properties']['segments'][0]['steps']
        geometry = route_data['features'][0]['geometry']['coordinates']
        
        for step in steps:
            instruction = step['instruction']
            instruction = instruction.replace("Turn", "Take a turn")
            instruction = instruction.replace("Continue", "Keep going")
            instruction = instruction.replace("Head", "Go")
            step['instruction'] = instruction
        
        return steps, geometry
        
    except Exception as e:
        print("Unable to calculate route")
        return None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def track_and_provide_directions(steps, geometry, destination_lat, destination_lon, tts_queue):
    destination_reached = False
    
    print("Starting navigation")
    
    try:
        while not destination_reached:
            current_position = get_current_location()
            if not current_position:
                print("Lost location signal. Please wait.")
                break
                
            current_lat, current_lon = current_position
            
            distance_to_destination = haversine(current_lat, current_lon, destination_lat, destination_lon)
            if distance_to_destination < 10:  
                speak("You have reached your destination.")
                break
            
            closest_step = None
            closest_distance = float('inf')
            
            for step in steps:
                start_index = step['way_points'][0]
                end_index = step['way_points'][1]
                step_coords = geometry[start_index:end_index + 1]
                step_end_lat, step_end_lon = step_coords[-1][1], step_coords[-1][0]
                
                distance = haversine(current_lat, current_lon, step_end_lat, step_end_lon)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_step = step
            
            if closest_step:
                instruction = closest_step['instruction']
                num_steps = meters_to_steps(closest_distance)
                steps_text = format_steps(num_steps)
                
                direction_text = f"{instruction}. Continue for {steps_text}."
                speak(direction_text)
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("Navigation stopped")
    except Exception as e:
        print("Navigation error occurred")

def compute_weights_and_location(classIds, bbox, confs, frame_width, frame_height):
    weights = []
    location_x = []
    location_y = []
    w_location = 0.8
    w_prediction = 0.5
    w_depth = 0.1

    c_dst_w = frame_width // 2
    c_dst_h = frame_height // 2

    for i, box in enumerate(bbox):
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        c_box_w = x1 + (x2 - x1) / 2
        c_box_h = y1 + (y2 - y1) / 2

        d1 = x1
        d2 = x2 - x1
        d3 = frame_width - x2

        v_depth = d2 / frame_width
        denominator = abs(d3 - d1)
        if denominator <= 0.01:
            v_loc = 1.0
        else:
            v_loc = min(1.0, d2 / denominator)
        weight = w_depth * v_depth + w_location * v_loc + w_prediction * confs[i]  

        h_val_cond = c_box_h - c_dst_h - 56
        w_val_cond = c_box_w - c_dst_w - 101

        if h_val_cond < 0:
            location_y.append((weight, "t"))
        else:
            location_y.append((weight, "b"))

        if w_val_cond > 0:
            location_x.append((weight, "r"))
        else:
            location_x.append((weight, "l"))

        weights.append((weight, classIds[i], (x1, y1, w, h)))

    location_x.sort(key=lambda x: (x[0], x[1]))
    location_y.sort(key=lambda x: (x[0], x[1]))
    weights.sort(key=lambda x: (x[0], x[1]))

    return weights, location_x, location_y

def visualize_and_analyze(image, classIds, bbox, frame_width, frame_height, classNames, confs, tts_queue, last_detection):
    weights, location_x, location_y = compute_weights_and_location(classIds, bbox, confs, frame_width, frame_height)
    current_time = time.time()
    
    # Draw bounding boxes for all detected objects
    for i, box in enumerate(bbox):
        x1, y1, w, h = box
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        label = classNames[classIds[i] - 1]
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if weights:
        h_val_cond = location_y[-1][0] if location_y else 0
        w_val_cond = location_x[-1][0] if location_x else 0
        bbox = weights[-1][2]
        
        # Highlight the most significant detection
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        cv2.rectangle(image, start_point, end_point, (255, 255, 0), 4)

        label = classNames[weights[-1][1] - 1]
        
        # Determine object position
        if abs(h_val_cond) < 50 and abs(w_val_cond) < 50:
            message = f"Center {label}"
        else:
            message = f"{location_x[-1][1]}{location_y[-1][1]} {label}"
        
        # Only add to queue if it's a new detection or significant time has passed
        if message != last_detection['message'] or \
           (current_time - last_detection['time']) >= 2.0:  # 2 second threshold
            if len(tts_queue) < 3:  # Limit queue size
                tts_queue.append(message)
            last_detection['message'] = message
            last_detection['time'] = current_time

    return image

def speak_text(tts_queue, stop_event):
    last_message = ""
    last_message_time = 0
    min_delay = 2  # Minimum delay between same messages in seconds
    
    while not stop_event.is_set():
        if tts_queue:
            message = tts_queue[0]  # Peek at the message without removing it
            current_time = time.time()
            
            # Only speak if it's a new message or enough time has passed
            if message != last_message or (current_time - last_message_time) >= min_delay:
                subprocess.call(['espeak', message])
                last_message = message
                last_message_time = current_time
                tts_queue.popleft()  # Remove the message after speaking
        time.sleep(0.1)  # Add small delay to prevent CPU overuse

def capture_camera_and_detect(tts_queue, stop_event):
    thres = 0.45
    nms_threshold = 0.2
    cap = cv2.VideoCapture(1)
    
    # Initialize last detection tracking
    last_detection = {
        'message': '',
        'time': 0
    }

    classNames = []
    classFile = "models/coco.names"
    with open(classFile, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")
    
    configPath = "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "models/frozen_inference_graph.pb"
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    
    prevTime = 0
    
    while not stop_event.is_set():
        success, img = cap.read()
        if not success:
            continue
            
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        
        if len(classIds) > 0:  # Only process if objects are detected
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))

            indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

            classIds = [classIds[i] for i in indices]
            bbox = [bbox[i] for i in indices]

            frame_height, frame_width = img.shape[:2]
            annotated_frame = visualize_and_analyze(
                img, classIds, bbox, frame_width, frame_height, 
                classNames, confs,  tts_queue, last_detection
            )
            
            # Calculate and display FPS
            cTime = time.time()
            fps = 1 / (cTime - prevTime)
            prevTime = cTime
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow("Live Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    try:
        speak("Welcome to blind navigation assistant")
        print("Initializing navigation system...")
        
        KUPONDOLE_LAT = 27.6891
        KUPONDOLE_LON = 85.3167
        
        origin = get_current_location()
        if origin:
            print("Calculating route to Kupondole...")
            steps, geometry = get_route_instructions(origin[0], origin[1], KUPONDOLE_LAT, KUPONDOLE_LON)
            
            if steps and geometry:
                print("Starting navigation...")
                # Use deque instead of list for thread-safe queue
                tts_queue = deque(maxlen=3)
                # Add stop event for clean thread termination
                stop_event = threading.Event()
                
                camera_thread = threading.Thread(
                    target=capture_camera_and_detect, 
                    args=(tts_queue, stop_event)
                )
                speak_thread = threading.Thread(
                    target=speak_text, 
                    args=(tts_queue, stop_event)
                )

                camera_thread.start()
                speak_thread.start()

                try:
                    track_and_provide_directions(steps, geometry, KUPONDOLE_LAT, KUPONDOLE_LON, tts_queue)
                finally:
                    stop_event.set()
                    camera_thread.join()
                    speak_thread.join()
            else:
                print("Unable to calculate route")
        else:
            print("Could not determine your location")
            
    except KeyboardInterrupt:
        speak("Program terminated")
    except Exception as e:
        speak("An error occurred")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()