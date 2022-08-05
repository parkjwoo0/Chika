## collector.py
- Input: Webcam
- Parameter
    - hand: "Right" or "Left"
    - session: num of sessions (int, 1~)
    - study_frame: num of frames (int, default 500) 
    - pause_time: when region changed (int, default 5) 
- Ouput: numpy array of position, velocity and acceleration 
    - position: (study_frame, 3, 21) 
    - velocity: (study_frame-1, 3, 21) 
    - acceleration: (study_frame-2, 3, 21) 
- Usage
```
python collector.py --hand [hand] --session [session] --study_frame [study_frame] -- pause_time [pause_time]
```

## visualize.ipynb
- You can compare the brushing teeth motion of two reions by data_type (velocity, acceleration).
