import cv2  
import numpy as np  
import time  
import datetime
import os
import matplotlib.pyplot as plt
import csv

signal_timing = 10

def store_vehicle_count(vehicle_count):
    """
    This function will store the final vehicle count and current system time in the given file. 
    It will create the file if it does not exist and append to the file if it exists.

    Parameters:
    file_path (str): The path of the file to store the data
    vehicle_count (int): The final vehicle count
    """
    # Get the current file path 
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the file path
    file_path = os.path.join(current_path, "vehicle_counter.csv")

    # Get current date and time
    now = datetime.datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

    # Open the file in append mode
    with open(file_path, "a") as f:
        # If file is empty, add header
        if f.tell() == 0:  
            f.write("Date, Time, Vehicle Count\n")

        # Append data to file 
        f.write(f"{date_time}, {vehicle_count}\n")

#web camera  
cap = cv2.VideoCapture('video.mp4')  
min_width_react = 80 #min width rectangle  
min_height_react = 80  #min height rectangle  
# count_line_position = 550 
 
# Get video height and width
height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)  

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
count_line_position = int(height - 50)

#Initialize Subtractor  
algo = cv2.createBackgroundSubtractorMOG2()  

def center_handle(x, y, w, h):  
    x1 = int(w/2)  
    y1 = int(h/2)  
    cx = x + x1  
    cy = y + y1  
    return cx, cy  

detect = []  
offset = 6 #Allowable error between pixels  
counter = 0  
temp_counter = 0
signal_timing = 10 #initial signal timing  

#Variables for speed detection
prev_frame_time = 0
new_frame_time = 0

while True:  
    ret, frame1 = cap.read()

    if not ret:  
        break  
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  
    blur = cv2.GaussianBlur(grey, (3, 3), 5)  
    
    # applying on each frame  
    img_sub = algo.apply(blur)  
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)  
    counterSahpe, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
    cv2.line(frame1, (25, count_line_position), (int(width), count_line_position), (255, 127, 0), 3)
    #cv2.line(frame1, (25, count_line_position), (width, count_line_position), (255, 127, 0), 3)  
    
    for (i, c) in enumerate(counterSahpe):  
        (x, y, w, h) = cv2.boundingRect(c)  
        validate_counter = (w >= min_width_react) and (h >= min_height_react)  
        if not validate_counter:  
            continue  
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)  
        cv2.putText(frame1, "Vehicle "+str(counter), (x, y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)  
        center = center_handle(x, y, w, h)  
        detect.append(center)  
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)  
    
    for (x, y) in detect:  
        if y < (count_line_position + offset) and y > (count_line_position - offset):  
            counter += 1 
            temp_counter = temp_counter + 1
        cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)  
        detect.remove((x, y))  
        print("Vehicle Counter: "+str(counter))
        print("temp Counter: "+str(temp_counter))  
    
    if temp_counter>signal_timing: #if vehicle count increases then increase signal timing  
        signal_timing+=5  
    
    if temp_counter >= 30:
        signal_timing = 0
        temp_counter = 0
    
    if signal_timing == 0:
        signal_timing = 10
         

    #Speed Detection
    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    speed = fps*2.236936 #2.236936 is the constant for mph
    cv2.putText(frame1, "Speed: "+str(int(speed))+" mph", (450, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)  
    
    cv2.putText(frame1, "VEHICLE COUNTER: "+str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)  
    cv2.putText(frame1, "Signal Timing: "+str(signal_timing), (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)  
    
    cv2.imshow('Video Original', frame1)  
    if cv2.waitKey(1) == 13:  
        break  
    
    # Add a delay of 100 milliseconds (0.01 seconds)  
    time.sleep(0.00001)  
    
store_vehicle_count(counter)

vehicle_count_list = [] 
time_list = []
with open('vehicle_counter.csv', newline='') as f: 
    reader = csv.reader(f) 
    for row in reader: 
        try:
            vehicle_count_list.append(int(row[2])) 
            time_list.append(row[1]) # Append time instead of index
        except ValueError: 
            pass # Skip this row

# Plot the graph
plt.plot(time_list, vehicle_count_list, 'r-')
plt.xlabel('Time')
plt.ylabel('Vehicle Count')
plt.title('Vehicle Count vs Time')

fig = plt.gcf()
fig.set_size_inches(12, 6)
fig.savefig('vehicle_count.png', dpi=300)
plt.show()

cv2.destroyAllWindows()  
cap.release()