from multiprocessing import Pool

import praw
from prawcore import Forbidden, NotFound
from praw.exceptions import ClientException
import numpy as np
import pandas as pd

from pymongo import MongoClient, errors

from bson.json_util import loads, dumps
from bson.objectid import ObjectId

from datetime import datetime as dt

import time

# TODO: move to app config file
CREDS_FILE = '/opt/cap1/.cap1'
MONGO_HOST = 'localhost'
MONGO_PORT = 27017
MONGO_DB = 'cap2'
MONGO_COLL = 'author'

def save_to_mongo(table, d):
    for key in ['_reddit']:
        _ = d.pop(key, None)
    try:
        table.insert_one(d)
    # TODO: check for unique index on id in collection
    except (errors.DuplicateKeyError, errors.InvalidDocument):
        pass

def get_author(reddit, author_fullname):
    author = reddit.redditor(fullname=author_fullname) 
    try:
        author._fetch()
        return dict(vars(author))
    except (Forbidden, NotFound):
        pass

def clean_author(d):
    for key in ['_reddit']:
        _ = d.pop(key, None)
    return d

def job(author_fullname):
    table = conn_mongo()
    reddit = conn_reddit(load_credentials(CREDS_FILE))
    raw_d = get_author(reddit, author_fullname)
    d = clean_author(raw_d)
    save_to_mongo(table, d)

def load_credentials(filename):
    with open(filename, 'r') as fp:
        creds = {}
        for line in fp:
            k, v = line.replace('\n','').split('\t')
            creds[k] = v
        return creds

def conn_mongo():
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[MONGO_DB]
    return db[MONGO_COLL]

def conn_reddit(creds):
    return praw.Reddit(client_id=creds['REDDIT_ID'],
        client_secret=creds['REDDIT_SECRET'],
        password=creds['REDDIT_PASSWORD'],
        username=creds['REDDIT_USERNAME'],
        user_agent='accessAPI:v0.0.1 (by /u/{})'.format(creds['REDDIT_USERNAME']))

def load_authors(filename):
    with open(filename, 'r') as fp:
        return fp.read().split('\n')

if __name__ == '__main__':

    table = conn_mongo()
    reddit = conn_reddit(load_credentials(CREDS_FILE))

    dataset = load_authors('data/unique_author_fullname')[4675:]
    for a in dataset:
        raw_d = get_author(reddit, a)
        if raw_d:
            d = clean_author(raw_d)
            save_to_mongo(table, d)
        else:
            print(f'did not receive redditor object for author {a}') 

    # Run this with a pool of 5 agents having a chunksize of 3 until finished
#    agents = 1
#    chunksize = 100
#    p = Pool(agents)
#    p.map(job, dataset, chunksize=chunksize)
#    p.close()
#    p.join()
