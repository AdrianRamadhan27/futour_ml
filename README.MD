# FuTour ML
Machine learning of FuTour application.


## Code Descriptions
0. "json/osm_data.json"
    The data was pulled from OpenStreetMap API (OSM API) as json file, which doesn't necessarily has the "genres" tag.
    However the json file itself contains the key "tourism" which represents the type of tourism such as village, attraction, and others.
    This will be used as the placeholder for the time being as this app is still a prototype.

1. geofetch.py
    To fetch images from Foursquare API and create json file that contains it.
    In this case, since the "osm_data_with_images.json" was already created, there's no need for geofecth to be ran again.

2. server.py
    To run server that launches web application from "templates/index.html".
    The web app is used to show the result of "osm_data_with_images.json" only.
    As for the images itselves, as it was pulled from Foursquare API, some might not be related to the places provided.
    Just think it as a placeholder images (prototype).

3. collaborative_filtering.py
    Collaborative filtering model for user rating data.

4. search_recommendation.py
    Search based recommendation model that requires manual querying (not yet NLP implemented).


## Running Apps
1. Installing dependencies (Note: you don't need all the dependencies, only the key ones like sklearn, tensorflow, pandas, and flask to run the web app)
    ```bash
    pip install requirements.txt
    ```

2. Running Web App
    To start the web app, run the following:
    ```bash
    python server.py
    ```
    Then input the IP and port that the server was running on to you browser.

3. Creating TensorFlow Model for Collaborative Filtering of Users Rating
    To create the model, run:
    ```bash
    python collaborative_filtering.py
    ```
    The model will automatically be created.

4. Implementing Search Recommendation
    To implement the search recommendation, we first should use the NLP model to read prompts from user and then generate queries that filters out the genre, then and only then we can input the queries to search_recommendation.py. After inputing queries from NLP, we can obtain the recommendation from the result and pass the result directly to the mobile application.

    search_recommendation.py requires a csv file that contains the places name and its rating. Hence, after the collaborative filtering model was fed, the ratings should be passed and appended to the "osm_data_with_images.json" file, which requires another code to append it and returns it as csv file, and this has yet to be done.