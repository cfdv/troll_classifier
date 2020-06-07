from pymongo import MongoClient

CREDS_FILE = '/opt/cap1/.cap1'
SAVED_MODELS = 'data/saved_models/'
MONGO_HOST = 'localhost'
MONGO_PORT = 27017
MONGO_DB = 'cap2'
MONGO_AUTHOR = 'author'
MONGO_COMMENT = 'comment'
MONGO_SUBMISSION = 'submission'

def load_credentials(filename):
    with open(filename, 'r') as fp:
        creds = {}
        for line in fp:
            k, v = line.replace('\n','').split('\t')
            creds[k] = v
        return creds

def conn_mongo(coll=None):
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[MONGO_DB]
    if coll:
        return db[coll]
    else:
        return db

def conn_reddit():
    creds = load_credentials(CREDS_FILE)
    return praw.Reddit(client_id=creds['REDDIT_ID'],
        client_secret=creds['REDDIT_SECRET'],
        password=creds['REDDIT_PASSWORD'],
        username=creds['REDDIT_USERNAME'],
        user_agent='accessAPI:v0.0.1 (by /u/{})'.format(creds['REDDIT_USERNAME']))

if __name__ == '__main__':
    pass