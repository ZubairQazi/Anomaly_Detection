import asyncpraw
import json
import zstandard as zstd
import datetime
import os

import openai

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

import string

from langdetect import detect

import asyncio
import aiofiles

# Gather credentials from config
with open('config.json') as f:
    config = json.load(f)

client_id = config['reddit_client_id']
client_secret = config['reddit_client_secret']
user_agent = config['reddit_user_agent']
username = config['reddit_username']
password = config['reddit_password']
openai_api_key = config['openai_api_key']

# Authenticate OpenAI API
openai.api_key = openai_api_key

time_filter = 'all'  # Can be one of: 'hour', 'day', 'week', 'month', 'year', 'all'

# Set the number of posts to grab
num_posts = 100
# Set the number of comments to retrieve
k = 10

# Convert date to ignore data after to Unix timestamp
date_obj = datetime.datetime.strptime('2022-10-31', "%Y-%m-%d")
unix_timestamp = int(date_obj.timestamp())

# Define the subreddits you want to scrape
subreddit_names = input('Enter subreddits (space separated): ').split()
# explainlikeimfive askscience askphilosophy

# Set the maximum number of requests allowed per minute
max_requests_per_minute = 3000

# Counter for tracking the number of requests made
request_counter = 0


async def process_post(post):
    global request_counter

    await post.load()

    post_data = {
        'title': post.title,
        'score': post.score,
        'subreddit': post.subreddit.display_name,
        'gpt': '',
        'comments': []
    }

    retry_attempts = 5
    for _ in range(retry_attempts):
        try:
            if request_counter >= max_requests_per_minute:
                print("Reached maximum requests per minute. Waiting for 1 minute...")
                await asyncio.sleep(60)
                request_counter = 0

            # Generate response using GPT-3.5 API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a frequent user of the subreddits {' '.join(subreddit_names)}. Answer anything relevant."},
                    {"role": "user", "content": post.title}
                ],
                temperature=0.7,
                max_tokens=100
            )
            generated_response = response.choices[0].message.content

            sentences = sent_tokenize(generated_response)
            complete_sentences = [sentence for sentence in sentences if sentence.endswith('.')]
            post_data['gpt'] = ' '.join(complete_sentences)

            await post.comments.replace_more(limit=0)
            comments = post.comments.list()
            comments_sorted = sorted(comments, key=lambda comment: getattr(comment, 'score', 0), reverse=True)

            for comment in comments_sorted[:k]:
                if detect(comment.body) == 'en' and comment.author is not None and len(comment.body) > 1:
                    comment_data = {
                        'score': comment.score,
                        'body': comment.body
                    }
                    post_data['comments'].append(comment_data)

            request_counter += 1

            # If everything succeeded, break out of the retry loop
            return post_data

        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded. Waiting for 1 minute...")
            await asyncio.sleep(60)
        except openai.error.APIError as e:
            print(f"Error occurred: {e.error_code}. Retrying in 1 minute...")
            await asyncio.sleep(60)

    else:
        print(f"Exceeded maximum retry attempts for post: {post.title}")


async def process_posts(posts):
    results = []

    for post in posts:
        post_data = await process_post(post)
        if post_data is not None:
            results.append(post_data)

    return results


async def process_subreddit(subreddit):
    top_posts = []
    async for post in subreddit.top(time_filter=time_filter, limit=num_posts):
        top_posts.append(post)
        # await asyncio.sleep(0.05)
    return top_posts


async def write_data_to_file(file_path, data):
    compressed_data = zstd.compress(json.dumps(data).encode('utf-8'))

    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(compressed_data)


async def main():

    # Authenticate using your Reddit account credentials
    reddit = asyncpraw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password, 
    )

    subreddits = [await reddit.subreddit(name) for name in subreddit_names]

    tasks = []
    for subreddit in subreddits:
        task = asyncio.create_task(process_subreddit(subreddit))
        tasks.append(task)

    top_posts = []
    for task in asyncio.as_completed(tasks):
        result = await task
        top_posts.extend(result)

    print(f'\nGathering top {num_posts} posts that satisfy criteria...')

    # Get the top posts that satisfy criteria below
    filtered_posts = [post for post in top_posts if post.score > 1000
                      and post.created_utc < unix_timestamp
                      and not post.over_18
                      # and '?' in post.title \
                      and detect(post.title) == 'en'
                      and not post.author is not None
                      and len(post.title) > 1]
    
    print(f'\nFiltered out {len(top_posts) - len(filtered_posts)} posts from {len(top_posts)} original posts')

    print(f'\nRetrieving top {k} comments from each post...')

    # Process the posts and retrieve the data
    results = await process_posts(filtered_posts)

    # Create dataset folder if it doesn't exist
    if not os.path.exists('datasets/reddit_datasets'):
        os.makedirs('datasets/reddit_datasets')

    print('\nWriting data to compressed file...')
    subreddit_name_string = '+'.join(subreddit_names)
    file_path = f'datasets/reddit_datasets/{subreddit_name_string}_{date_obj.date()}_top-{k}-comments_json.zst'
    await write_data_to_file(file_path, results)

    await reddit.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {str(e)}")

