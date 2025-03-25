import os
import warnings
import argparse
import cv2
import numpy as np
import time
import requests  # Added for sending HTTP requests
from math import ceil
from utils.default_config import _C
from engine import MaskRCNNPredictor, DMRCNNPredictor


warnings.filterwarnings('ignore')


ESP_IP = "http://192.168.1.5" # ESP8266 IP from coner.py


parser = argparse.ArgumentParser(description="Predictor")


parser.add_argument('--config_file', type=str, required=True, help="Path of config file (.yaml)")
parser.add_argument('--weight_file', type=str, required=True, help="Path of weight file (.pth)")
parser.add_argument('--conf_score', type=float, default=0.6, help="Confidence threshold for inference")


parser.add_argument('--image_file', type=str, default='', help="Path of single image file")
parser.add_argument('--image_dir', type=str, default='', help="Directory which contains multiple image files")
parser.add_argument('--save_dir', type=str, default='')


parser.add_argument('--input_size', type=int, default=800, help="Determine the size of the image to be used for inference.")


# Function to send commands to ESP8266
def send_command(command):
   try:
       response = requests.get(f"{ESP_IP}/message?text={command}")
       print(f"Sent '{command}' command to ESP8266")
       print("Response from ESP8266:", response.text)
   except requests.exceptions.RequestException as e:
       print(f"Error sending command: {e}")


def calculate_greenlight_time(vehicle_count):
   x = ceil((vehicle_count * 2) / 10) * 10
   return max(10, min(x, 60))


def select_road_areas_by_points(image_path, n):
   image = cv2.imread(image_path)
   if image is None:
       print("Error: Unable to read the image.")
       return []


   areas = []
   temp_area = []
   window_name = "Select Road Areas (Press 'n' to finalize area, 'q' to quit)"


   def mouse_callback(event, x, y, flags, param):
       nonlocal temp_area
       if event == cv2.EVENT_LBUTTONDOWN:
           temp_area.append((x, y))
           print(f"Point added: ({x}, {y})")
           cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
           cv2.imshow(window_name, image)


   cv2.imshow(window_name, image)
   cv2.setMouseCallback(window_name, mouse_callback)


   while len(areas) < n:
       key = cv2.waitKey(0) & 0xFF
       if key == ord('n'):  # Finalize current area
           if len(temp_area) > 2:  # At least 3 points needed for a polygon
               areas.append(temp_area.copy())
               print(f"Area {len(areas)} finalized: {temp_area}")
               temp_area = []
           else:
               print("An area must have at least 3 points.")
       elif key == ord('q'):  # Quit early
           print("Exiting selection process early.")
           break


   cv2.destroyAllWindows()
   return areas


def is_point_inside_polygon(point, polygon):
   x, y = point
   n = len(polygon)
   inside = False


   px, py = polygon[0]
   for i in range(1, n + 1):
       sx, sy = polygon[i % n]
       if y > min(py, sy):
           if y <= max(py, sy):
               if x <= max(px, sx):
                   if py != sy:
                       xinters = (y - py) * (sx - px) / (sy - py) + px
                   if px == sx or x <= xinters:
                       inside = not inside
       px, py = sx, sy


   return inside


def count_vehicles_in_area(vehicle_locations, areas):
   count = 0
   for vehicle in vehicle_locations:
       vehicle_center = ((vehicle["x1"] + vehicle["x2"]) / 2, (vehicle["y1"] + vehicle["y2"]) / 2)
       if any(is_point_inside_polygon(vehicle_center, area) for area in areas):
           count += 1
   return count


if __name__ == "__main__":
   args = parser.parse_args()


   # Check Config
   cfg = _C.clone()
   cfg.set_new_allowed(True)
   cfg.merge_from_file(args.config_file)


   cfg.MODEL.DEVICE = 'cpu'
   cfg.INPUT.RESIZE = args.input_size
   print(f"Device being used: {cfg.MODEL.DEVICE}")


   predictor_cls = DMRCNNPredictor if cfg.MODEL.N2V.USE else MaskRCNNPredictor
   predictor = predictor_cls(args.config_file, args.weight_file, args.conf_score)


   if len(args.image_dir) > 0:
       assert os.path.isdir(args.image_dir), "Cannot find 'image_dir' you entered."


       # Sort images numerically
       image_list = sorted(
           [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))],
           key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
       )


       print(f"Found {len(image_list)} images from '{args.image_dir}' for the inference.")
       num_roads = int(input("Enter the number of roads in the junction: "))
       first_image_path = image_list[0]
      
       road_areas = select_road_areas_by_points(first_image_path, num_roads)


       image_idx = 0


       while image_idx < len(image_list):
           for road_idx in range(num_roads):
               if image_idx >= len(image_list):
                   break


               current_image_path = image_list[image_idx]
               print(f"Processing {current_image_path} for road {road_idx + 1}...")


               img_arr = cv2.imread(current_image_path)
               vehicle_locations = predictor.get_vehicle_locations(img_arr)


               vehicle_count = count_vehicles_in_area(vehicle_locations, [road_areas[road_idx]])
               greenlight_time = calculate_greenlight_time(vehicle_count)


               scaled_greenlight_time = greenlight_time // 10


               print(f"{road_idx + 1}_ON time={scaled_greenlight_time}sec")
               print(f"Vehicles detected in road_{road_idx + 1} area: {vehicle_count}")


               if scaled_greenlight_time > 0:
                   send_command(f"{road_idx + 1}_ON")
                   time.sleep(scaled_greenlight_time)
               image_idx += greenlight_time // 10
