# Blackjack Counter
This dataset was taken from a Roboflow and trained via Google Colab on Yolov8

## 1. Summary 

##### Overview
Card counting is a known blackjack strategy to have the best odds of beating or breaking even against the house.

At a glance, the strategy is just to keep track of low cards (2-6) which is asigned a value of +1, neutral or midding cards (7,8,9) +0, and high cards/aces (10, J, Q, K, A) -1. This allows you to know how many high cards are left in the deck which is advantages to the player and generally worse for the house. This combined with basic blackjack strategy gives a player a huge edge. So for example, if the count is +5, in a deck of 52 cards, then the chances of a high card coming off the top is extremely high, since a lot of low cards (2-6) have been used up. 

Build a object detection model (Yolov8) to keep the card "count" in Blackjack.

##### Technical Overview
I used Google Colab to train a small Yolov8 model to detect cards. I then utilized the framework provided by Ultralytics / cv2 to annotate and perform object tracking. 
The model needed more training to take advantage of object tracking, however due to the lack of resources, I limited counting to a single deck.

## 2. Results
After using a dataset with ~10000 images playing cards, I was able to create a modest card detection on a Yolov8-s model. It is able to detect cards fairly accurately when one card was shown at a time. However, I experienced weird behavior when multiple cards are shown at a time next to each other. I believe this is due to the cards being fairly close and the low resolution video. 

The next steps for this project would be training a bigger model and use more curated images. This would then open the doors for object tracking, which will allow for card counting using multiple decks.

## 3. Data Description
10000 images of cards from RoboFlow
  
## 4. Libraries
- pytorch
- opencv
- numpy
- pandas
- matplotlib
- ultralytics
- supervision


## 5. References
ultralytics
Roboflow


