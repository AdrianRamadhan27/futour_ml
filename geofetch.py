import json
import requests
import time

# Load GeoJSON data
with open('./json/osm_data.geojson', 'r', encoding='utf-8') as file:
    geojson_data = json.load(file)

# Extract relevant fields from GeoJSON
locations = []
for feature in geojson_data['features']:
    properties = feature['properties']
    geometry = feature['geometry']
    if 'name' in properties:
        locations.append({
            'name': properties['name'],
            'type': properties.get('tourism', 'unknown'),
            'latitude': geometry['coordinates'][1],
            'longitude': geometry['coordinates'][0]
        })

# Unsplash API settings
UNSPLASH_API_KEY = 'ow6rt7fcJ5EpUUFLOjKk1Uu_V_htagROG3fPwVwHkhU'
headers_unsplash = {"Authorization": f"Client-ID {UNSPLASH_API_KEY}"}
search_url_unsplash = "https://api.unsplash.com/search/photos"

# Function to fetch images with retry logic
def fetch_images(query, max_results=3, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            params = {"query": query, "per_page": max_results}
            response = requests.get(search_url_unsplash, headers=headers_unsplash, params=params)
            response.raise_for_status()  # Check if the response is OK (status code 200)
            
            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        "description": img.get("alt_description"),
                        "url": img["urls"]["small"]
                    }
                    for img in data["results"]
                ]
            else:
                print(f"Error fetching images for {query}: {response.status_code}")
                return []

        except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
            print(f"Error fetching images for {query}: {e}")
            attempt += 1
            print(f"Retrying... ({attempt}/{retries})")
            time.sleep(delay)  # Wait before retrying
            
    return []  # Return empty list if all retries fail

# # Foursquare API settings
# FOURSQUARE_API_KEY = 'XAB0PXWAMWUPXQYOEEZZHVTHRSD32HYVZA34OVZ3DH33SWJM'
# headers_foursquare = {'Authorization': FOURSQUARE_API_KEY}
# search_url_foursquare = 'https://api.foursquare.com/v3/places/search'

# # Function to fetch ratings with retry logic
# def fetch_ratings(location, retries=3, delay=5):
#     params = {
#         'query': location['name'],
#         'll': f"{location['latitude']},{location['longitude']}",
#         'limit': 1  # Fetch the most relevant result
#     }
#     attempt = 0
#     while attempt < retries:
#         try:
#             response = requests.get(search_url_foursquare, headers=headers_foursquare, params=params)
#             response.raise_for_status()  # Check if the response is OK (status code 200)
            
#             if response.status_code == 200:
#                 data = response.json()
#                 if data['results']:
#                     return data['results'][0].get('rating', 'No rating available')
#             return 'No rating available'
        
#         except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
#             print(f"Error fetching rating for {location['name']}: {e}")
#             attempt += 1
#             print(f"Retrying... ({attempt}/{retries})")
#             time.sleep(delay)  # Wait before retrying

#     return 'No rating available'

# Pairing images and ratings with locations
for location in locations:
    images = fetch_images(location['name'])
    if not images:  # Fallback to type if no images found
        images = fetch_images(location['type'])
    location['images'] = images
    # print(f"Fetched data for {location['name']} - Rating: {location['rating']}, Images: {len(images)}")
    print(f"Fetched data for {location['name']}, Images: {len(images)}")

# Update GeoJSON with ratings and images
for feature in geojson_data['features']:
    properties = feature['properties']
    for location in locations:
        if properties.get('name') == location['name']:
            properties['images'] = [img['url'] for img in location.get('images', [])]
            # properties['rating'] = location['rating']

# Save updated GeoJSON with images and ratings
with open('json/osm_data_with_images.geojson', 'w', encoding='utf-8') as file:
    json.dump(geojson_data, file, indent=4)

print("GeoJSON updated with image URLs!")
