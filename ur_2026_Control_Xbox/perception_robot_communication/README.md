# CV and UR5 Script Communication

This folder contains template code from a previous 2.12/2.120 term project. It is provided as one possible way to initiate communication between a perception stack and a UR5 stack.

The communication method used here is simple:

- the perception script writes a JSON object to `CV_pick_place_data.json`
- the UR5 script reads that JSON file and uses the values inside it

The two scripts therefore work as a loosely coupled producer/consumer pair. The relevant parts of the code that relates to the communication is shown below:

- `CV_pick_place.py` detects bottles on a conveyor, estimates target motion, and writes the relevant data to the shared JSON file.

```python
#define what data is necessary and to be stored in the json
data = {
    "target_pos": moving_avg_pos,
    "target_vel": moving_avg_vel,
    "target_pos2": moving_avg_pos2,
    "vel_reset": vel_reset,
}

with open("CV_pick_place_data.json", "w") as file:
    try:
        json.dump(data, file)
    except Exception as e:
        print("Error:", e, "data:", data)
```

- `UR5_pick_place.py` reads that data from the same JSON file and then uses it for robot-side decision making.

```python
#open and load the json file
with open("CV_pick_place_data.json", "r") as file:
    try:
        data = json.load(file)
    except Exception as e:
        print("Error:", e)
```


This is only one example of how communication between the two stacks can be implemented. There are multiple possible approaches, and each team should choose the method that best fits its task. 

For further interest in how this was integrated, feel free to read through the codes!
