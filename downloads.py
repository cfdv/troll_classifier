import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as BS
import re
from os import path
import wget

def parse_information_operations_html(
            html_file='data/twitter_information_operations.html'):
    '''
    I acquired the information operations html file manually after 
    giving my email address to the form here:
    https://transparency.twitter.com/en/information-operations.html
    
    returns pandas dataframe containing metadata for coordinated attacks 
        on elections as reported by twitter
    '''
    with open(html_file, 'r') as fp:
        html = fp.read()
    soup = BS(html, 'html.parser')
    
    accounts = [e.parent['href'] for e in soup.findAll(text=re.compile('Account '))]
    tweets = [e.parent['href'] for e in soup.findAll(text=re.compile('Tweet '))]
    media = [e.parent['href'] for e in soup.findAll(text=re.compile('Media'))]
    events_df = pd.DataFrame(parse_event_titles(soup.select('b')))
    events_df['accounts_url'] = accounts
    events_df['tweets_url'] = tweets
    events_df['media_url'] = media
    return events_df

def parse_event_titles(event_titles):
    '''
    Input: event_titles (list of bs4.element.Tag)
    Output: list of dict containing attributes for each event
    
    e.g. 
        [<b>Egypt (February  2020) - 2541 Accounts</b>]
    returns    
        [{'country': 'Egypt',
         'month': 'February',
         'year': '2020',
         'num_accounts': '2541'}]
    '''
    keys = ['country', 'month', 'year', 'num_accounts', 'set']
    events = []
    for t in event_titles:
        temp_t = t.string
        for mychars in ['\xa0', ' -', ',', ')']:
            temp_t = temp_t.replace(mychars, '')
        parsed = temp_t.split('(') # account for spaces in source of coordinated activity
        
        country = parsed[0].strip()
        other_attr = parsed[1].split(' ')[:-1]
        this_set = '1'
        if len(other_attr) > 4: # 'set' appears if multiple releases from same source
            this_set = other_attr[3] # e.g. Venezuela (January 2019, set 2) - 764 accounts
            other_attr = other_attr[:2] + [other_attr[-1]]
        stripped = [parsed[0].strip()] + other_attr + [this_set]
        events.append(dict(zip(keys, stripped)))
    return events

def download_from_google_cloud_storage(raw_url, 
                          target_local_dir='data/twitter', 
                          bucket='twitter-election-integrity'):
    '''
    given a raw link from twitter's election integrity page, download 
    file from google cloud storage to local data directory 
    
    input: raw_url (str) from twitter election integrity page
    return None
    
    e.g.
    'https://storage.cloud.google.com/twitter-election-integrity/hashed/2020_04/egypt_022020/egypt_022020_users_csv_hashed.zip'
        downloads
    'https://twitter-election-integrity.storage.googleapis.com/hashed/2020_04/egypt_022020/egypt_022020_users_csv_hashed.zip'
    '''
    twitter_google_cloud_storage = 'https://' + bucket + '.storage.googleapis.com'

    url_parts = raw_url.split('/')
    filename = url_parts[-1]
    local_file_path = target_local_dir + '/' + filename
    if path.exists(local_file_path):
        print(local_file_path + ' exists...')
        return None
    else:
        print(f'Downloading {raw_url} ...')
        target_url = '/'.join([twitter_google_cloud_storage] + url_parts[4:])
        #r = requests.get(target_url, allow_redirects=True)
        #with open(local_file_path, 'bw') as fp:
        #    fp.write(r.content)
        wget.download(target_url, local_file_path)

if __name__ == '__main__':
    events_df = parse_information_operations_html()
    for col in ['accounts_url', 'tweets_url']:
        for raw_url in events_df[col]:
            download_from_google_cloud_storage(raw_url)
    