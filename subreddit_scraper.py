import praw
import json
import zstandard as zstd
import datetime
import os

import openai

# Gather credentials from config
with open('config.json') as f:
    config = json.load(f)

client_id = config['reddit_client_id']
client_secret = config['reddit_client_secret']
user_agent = config['reddit_user_agent']
username = config['reddit_username']
password = config['reddit_password']
openai_api_key = config['openai_api_key']

# Authenticate using your Reddit account credentials
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent,
                     username=username,
                     password=password)

# Authenticate OpenAI API
openai.api_key = openai_api_key

# explainlikeimfive askscience askphilosophy

time_filter = 'all'  # Can be one of: 'hour', 'day', 'week', 'month', 'year', 'all'

# Set the number of posts to grab
num_posts = 100
# Set the number of comments to retreive
k = 5

# Convert date to ignore data after to Unix timestamp
date_obj = datetime.datetime.strptime('2022-10-31', "%Y-%m-%d")
unix_timestamp = int(date_obj.timestamp())

# Define the subreddits you want to scrape
subreddit_names = input('Enter subreddits (space separated): ').split()

# Get the subreddit objects
subreddits = [reddit.subreddit(name) for name in subreddit_names]

# Get the top posts that have more than 1000 upvotes and were created before October 2022
top_posts = []
for subreddit in subreddits:
    top_posts.extend(subreddit.top(time_filter=time_filter, limit=100))

print(f'\nGathering top {num_posts} posts that satisfy criteria...')

# Get the top posts that have more than 1000 upvotes
filtered_posts = [post for post in top_posts if post.score > 1000 \
                  and post.created_utc < unix_timestamp \
                    and not post.over_18 \
                        and '?' in post.title]


print(f'\nRetrieving top {k} comments from each post...')

# Retrieve the top 5 comments for each post
data = []

for post in filtered_posts:
    post_data = {
        'title': post.title,
        'score': post.score,
        'gpt': '',
        'comments': []
    }

    # Generate response using GPT-3.5 API
    prompt = f'Post: {post.title}\nResponse:'
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    generated_response = response.choices[0].text.strip()
    post_data['gpt'] = generated_response

    post.comments.replace_more(limit=0)
    comments = post.comments.list()
    comments_sorted = sorted(comments, key=lambda comment: comment.score, reverse=True)

    for comment in comments_sorted[:k]:
        comment_data = {
            'score': comment.score,
            'body': comment.body
        }

        post_data['comments'].append(comment_data)

    data.append(post_data)

# create dataset folder if it doesn't exist
if not os.path.exists('reddit_datasets'):
    os.makedirs('reddit_datasets')

print('\nWriting data to compressed file...')

# Compress the data and store it in a file
compressed_data = zstd.compress(json.dumps(data).encode('utf-8'))

subreddit_name_string = '+'.join(subreddit_names)
with open(f'reddit_datasets/{subreddit_name_string}_{date_obj.date()}_top-{k}-comments.zst', 'wb') as f:
    f.write(compressed_data)