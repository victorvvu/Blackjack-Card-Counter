import os
import random

import cv2
import ultralytics 



def main():
    
    #card counting lists
    plus_one = ['2D','2C','2H','2S','3D','3C','3H','3S','4D','4C','4H','4S',
                '5D','5C','5H','5S','6D','6C','6H','6S']
    minus_one = ['QC',  'QD',  'QH', 'QS','KC',  'KD',  'KH', 'KS','JC',  'JD',  'JH', 'JS',
                 'AC',  'AD',  'AH', 'AS','10C',  '10D',  '10H', '10S']
    plus_zero = ['7C',  '7D',  '7H', '7S','8C',  '8D',  '8H', '8S','9C',  '9D',  '9H', '9S']
    
    video= 'your path'
    
    cap = cv2.VideoCapture(video)
    model = ultralytics.YOLO('yolov8s_playing_cards.pt')
    
    CAP_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CAP_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    
    writer= cv2.VideoWriter('output', 
                                cv2.VideoWriter_fourcc(*'DIVX'), 
                                7, 
                                (CAP_WIDTH, CAP_HEIGHT))
    
    
    CLASS_NAMES_DICT = model.model.names
    counter =0
    seen_cards = []
    while True:
        ret, frame = cap.read() 
        if not ret:
            break
        results = model.track(frame, persist=True,conf=0.9)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            #Could use IDS to perform tracking 
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for c in results[0].boxes.cls:
                print((model.names[int(c)]))
                if model.names[int(c)] in plus_one:
                    plus_one.remove(model.names[int(c)])
                    seen_cards.append(model.names[int(c)])
                    counter += 1
                    
                if model.names[int(c)] in minus_one:
                    minus_one.remove(model.names[int(c)])
                    seen_cards.append(model.names[int(c)])
                    counter -= 1
                if model.names[int(c)] in plus_zero:
                    plus_zero.remove(model.names[int(c)])
                    seen_cards.append(model.names[int(c)])
                    
                    
                    
                    
            for box, id in zip(boxes, ids):
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{model.names[int(c)]}",
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                
                
        #COUNTER
        cv2.putText(
                        frame,
                        f"Counter: {counter}",
                        (0,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
        #cards seen
        cv2.putText(
                        frame,
                        f"Identified cards: {' '.join(seen_cards)}",
                        (0,CAP_HEIGHT-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .5,
                        (0, 255, 0),
                        2,
                    )
        writer.write(frame)
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print(seen_cards)
    print(counter)

if __name__ == "__main__":
    main()