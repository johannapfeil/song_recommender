import requests
from bs4 import BeautifulSoup
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import time
from tqdm import tqdm
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def scrape_billboard_hot_100():
    """
    Scrape the Billboard Hot 100 chart
    """
    url = "https://www.billboard.com/charts/hot-100/"

    # Prevent blocking, different content, or denied access
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    songs = []
    artists = []
    
    chart_items = soup.select('ul.o-chart-results-list-row')
    
    for item in chart_items:
        title = item.select_one('h3#title-of-a-story').text.strip()
        artist = item.select_one('span.c-label.a-no-trucate').text.strip()
        
        songs.append(title)
        artists.append(artist)
    
    df = pd.DataFrame({'Song': songs, 'Artist': artists})
    
    return df

def clean_million_song_subset(df):
    """
    Clean the million song subset dataframe
    """
    df['title'] = df['title'].str[2:-1]
    df['artist'] = df['artist'].str[2:-1]
    df = df.rename(columns={'title': 'Song', 'artist': 'Artist'})
    
    return df

def setup_spotify():
    """
    Initialize Spotify client
    """
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')

    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    return sp

def get_track_features(sp, track_name, artist_name):
    
    """Search for a track and get its audio features"""
    
    query = f"track:{track_name} artist:{artist_name}"
    results = sp.search(q=query, type='track', limit=1)
        
    if not results['tracks']['items']:
        return None
        
    track = results['tracks']['items'][0]
    
    artist_id = track['artists'][0]['id']
    artist_info = sp.artist(artist_id)
    
    track_info = {
            'track_id': track['id'],
            'popularity': track['popularity'],
            'duration_ms': track['duration_ms'],
            'album_release_year': track['album']['release_date'][:4],
            'album_cover_url': track['album']['images'][0]['url'],
            'explicit': track['explicit'],
            'artist_popularity': artist_info['popularity'],
            'artist_genres': artist_info['genres']}
            
    return track_info

def clean_artist_name(artist_name):
    """
    Clean the artist name by removing delimiters
    """
    delimiters = ['&', 'feat.', 'featuring', 'Featuring', 'FEATURING', 'Feat.', 'FEAT.', 'ft.', 'Ft.', 'FT.', '/', 'with', 'WITH', 'With', ',', 'And', 'and', 'And']
    for delimiter in delimiters:
        if delimiter in artist_name:
            artist_name = artist_name.split(delimiter)[0].strip()
    
    return artist_name

def main(df):

    sp = setup_spotify()

    current_batch = []
    failed_tracks = []
    batch_number = 51
    processed_count = 0

    output_dir = '../data/batches'

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing tracks"):
        
        if processed_count > 0 and processed_count % 50 == 0:
            print(f"\nPausing for 30 seconds after processing {processed_count} tracks...")
            time.sleep(30)

        clean_artist = clean_artist_name(row['Artist'])
        
        try:
            features = get_track_features(sp, track_name=row['Song'],artist_name=clean_artist)
            processed_count += 1

        except Exception as e:
            print(f"Error processing track {row['Song']} by {clean_artist}: {e}")
            processed_count += 1
            continue
        
        if features:
            row_dict = row.to_dict()
            row_dict.update(features)
            current_batch.append(row_dict)

            if len(current_batch) == 50 or index == len(df) - 1:
                if current_batch:
                    batch_number += 1
                    batch_df = pd.DataFrame(current_batch)
                    batch_filename = f'{output_dir}/songs_batch_{batch_number}.csv'
                    
                    batch_df.to_csv(batch_filename, index=False)
                    print(f"\nSaved batch {batch_number} with {len(current_batch)} tracks")
                    
                    current_batch = []
        else:
            failed_tracks.append({'track_name': row['Song'], 'artist_name': row['Artist']})
    
    if current_batch:
        batch_number += 1
        batch_df = pd.DataFrame(current_batch)
        batch_filename = f'{output_dir}/songs_batch_{batch_number}.csv'
        batch_df.to_csv(batch_filename, index=False)
        print(f"\nSaved final batch {batch_number} with {len(current_batch)} tracks")
        
    if failed_tracks:
        failed_df = pd.DataFrame(failed_tracks)
        failed_df.to_csv(f'{output_dir}/failed_tracks.csv', index=False)
        print(f"\nFailed to process {len(failed_tracks)} tracks (saved to failed_tracks.csv)")

def encode_genres(df, genre_column='artist_genres'):
    """
    Convert string representation of list to actual list
    """
    df[genre_column] = df[genre_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df[genre_column])
    genres_df = pd.DataFrame(
            genres_encoded,
            columns=[f'{genre}' for genre in mlb.classes_],
            index=df.index)
    
    result_df = pd.concat([df, genres_df], axis=1)
    
    return result_df