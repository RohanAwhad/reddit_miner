import asyncio
import dataclasses
import json
import openai
import os
import numpy as np
import pickle
import praw
import re

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Optional


@dataclasses.dataclass
class Message:
    role: str
    content: str

@dataclasses.dataclass
class Post:
    id: str
    title: str
    selftext: str
    n_upvotes: int
    n_comments: int
    comments: list[str]
    merged_content: str = ''
    pain_n_anger: Optional[list[str]] = None
    advice_requests: Optional[list[str]] = None
    solution_requests: Optional[list[str]] = None
    embeddings: Optional[list[float]] = None

@dataclasses.dataclass
class ClusterPost:
    id: str
    title: str
    n_upvotes: int
    n_comments: int
    merged_content: str
    highlighted_point: str

@dataclasses.dataclass
class Cluster:
    id: str
    posts: list[ClusterPost]
    n_upvotes: int  # total of all posts in this cluster
    n_comments: int
    n_posts: int
    title: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "posts": [post.__dict__ for post in self.posts],
            "n_upvotes": self.n_upvotes,
            "n_comments": self.n_comments,
            "n_posts": self.n_posts,
            "title": self.title
        }

    
N_CONCURRENT_LLM_CALL = 30
async def main(subreddit_list: list[str]) -> tuple[list[Cluster], list[Cluster], list[Cluster]]:
    # given a list of subreddits, use praw to get the hottest posts
    # Specifically, their title, body and comment threads, upto depth=1
    # Then merge the post contents into one single post called merged post
    # Use openai llm to extract top 3 pain n anger points
    # top-3 advice requests and top-3 solution requests for each merged post
    # embed each of those points and save them.
    # Using DBSCAN, cluster the pain n anger, advice requests and solution requests point
    # for each cluster, generate a new point as its title
    # return each cluster with their post id and title

    print('Gathering posts ...')
    hottest_posts: list[Post] = []
    for sr in subreddit_list: hottest_posts.extend(get_hot_posts_and_comments(sr, limit=20))


    print('Extracting insights ...')
    for i in tqdm(range(0, len(hottest_posts), N_CONCURRENT_LLM_CALL), desc='Extracting points:'):
        coros = []
        for post in hottest_posts[i:i+N_CONCURRENT_LLM_CALL]:
            post.merged_content = merge_content(post)
            coros.extend([
                get_pain_n_anger(post.merged_content),
                get_advice_requests(post.merged_content),
                get_solution_requests(post.merged_content),
            ])
        result = await asyncio.gather(*coros)
        for i, post in zip(range(0, len(result), 3), hottest_posts[i:i+N_CONCURRENT_LLM_CALL]):
            post.pain_n_anger = result[i]
            post.advice_requests = result[i+1]
            post.solution_requests = result[i+2]


    print('Generating embeddings')
    all_pain_n_anger_embeddings = []
    all_advice_requests_embeddings = []
    all_solution_requests_embeddings = []

    pna_idx2post_id, pna_idx2point_id = {}, {}
    ar_idx2post_id, ar_idx2point_id = {}, {}
    sa_idx2post_id, sa_idx2point_id = {}, {}
    for post in tqdm(hottest_posts, desc='Generating Embeddings:'):
        pain_n_anger_embeddings: Optional[list[list[float]]] = embed(post.pain_n_anger)  if post.pain_n_anger else None # returns batch embeddings for all pain n anger points
        advice_requests_embeddings: Optional[list[list[float]]] = embed(post.advice_requests) if post.advice_requests else None   # returns batch embeddings for all advice requests points
        solution_requests_embeddings: Optional[list[list[float]]] = embed(post.solution_requests) if post.solution_requests else None   # returns batch embeddings for all solution_requests points

        # save idx to post id for each embedding from the batch 
        if pain_n_anger_embeddings:
            for i, embedding in enumerate(pain_n_anger_embeddings):
                pna_idx2post_id[len(all_pain_n_anger_embeddings)] = post.id
                pna_idx2point_id[len(all_pain_n_anger_embeddings)] = i
                all_pain_n_anger_embeddings.append(embedding)

        if advice_requests_embeddings:
            for i, embedding in enumerate(advice_requests_embeddings):
                ar_idx2post_id[len(all_advice_requests_embeddings)] = post.id
                ar_idx2point_id[len(all_advice_requests_embeddings)] = i
                all_advice_requests_embeddings.append(embedding)

        if solution_requests_embeddings:
            for i, embedding in enumerate(solution_requests_embeddings):
                sa_idx2post_id[len(all_solution_requests_embeddings)] = post.id
                sa_idx2point_id[len(all_solution_requests_embeddings)] = i
                all_solution_requests_embeddings.append(embedding)


    print('Clustering ...')
    pain_n_anger_clusters: list[Cluster] = cluster(all_pain_n_anger_embeddings, pna_idx2post_id, pna_idx2point_id, hottest_posts, 'pain_n_anger')
    advice_requests_clusters: list[Cluster] = cluster(all_advice_requests_embeddings, ar_idx2post_id, ar_idx2point_id, hottest_posts, 'advice_requests')
    solution_requests_clusters: list[Cluster] = cluster(all_solution_requests_embeddings, sa_idx2post_id, sa_idx2point_id, hottest_posts, 'solution_requests')

    for i in range(0, len(pain_n_anger_clusters), N_CONCURRENT_LLM_CALL):
        coros = []
        for cl in pain_n_anger_clusters[i: i+N_CONCURRENT_LLM_CALL]:
            points_in_cl = [post.highlighted_point for post in cl.posts]
            combined_text = '\n'.join(points_in_cl)
            coros.append(generate_title(combined_text))

        result = await asyncio.gather(*coros)
        for res, cl in zip(result, pain_n_anger_clusters[i: i+N_CONCURRENT_LLM_CALL]): cl.title = res

    for i in range(0, len(advice_requests_clusters), N_CONCURRENT_LLM_CALL):
        coros = []
        for cl in advice_requests_clusters[i: i+N_CONCURRENT_LLM_CALL]:
            points_in_cl = [post.highlighted_point for post in cl.posts]
            combined_text = '\n'.join(points_in_cl)
            coros.append(generate_title(combined_text))

        result = await asyncio.gather(*coros)
        for res, cl in zip(result, advice_requests_clusters[i: i+N_CONCURRENT_LLM_CALL]): cl.title = res

    for i in range(0, len(solution_requests_clusters), N_CONCURRENT_LLM_CALL):
        coros = []
        for cl in solution_requests_clusters[i: i+N_CONCURRENT_LLM_CALL]:
            points_in_cl = [post.highlighted_point for post in cl.posts]
            combined_text = '\n'.join(points_in_cl)
            coros.append(generate_title(combined_text))

        result = await asyncio.gather(*coros)
        for res, cl in zip(result, solution_requests_clusters[i: i+N_CONCURRENT_LLM_CALL]): cl.title = res

    return (
        sorted(pain_n_anger_clusters, key=lambda x: x.n_upvotes, reverse=True),
        sorted(advice_requests_clusters, key=lambda x: x.n_upvotes, reverse=True), 
        sorted(solution_requests_clusters, key=lambda x: x.n_upvotes, reverse=True)
    )


def get_hot_posts_and_comments(subreddit_name, limit) -> list[Post]:
    global post_id

    reddit = praw.Reddit(client_id=os.environ['REDDIT_CLIENT_ID'],
                         client_secret=os.environ['REDDIT_API_KEY'],
                         user_agent='test-bot by u/ronny_500')

    subreddit = reddit.subreddit(subreddit_name)
    top_posts = subreddit.hot(limit=limit)
    
    result = []
    for post in top_posts:
        if post.stickied: continue
        comment_list = []
        n_upvotes = post.score
        for comment in post.comments.list():
            # check if MoreComment is present
            if isinstance(comment, praw.models.MoreComments): continue
            comment_list.append(comment.body)
            n_upvotes += comment.score
        
        result.append(Post(
            id = post.id,
            title=post.title,
            selftext=post.selftext,
            n_upvotes=n_upvotes,
            n_comments=len(comment_list),
            comments=comment_list
        ))

    return result

def merge_content(post: Post) -> str:
    """Merge post title, selftext, and comments into one string"""
    merged_text = post.title + '\n' + post.selftext + '\n'
    for comment in post.comments:
        merged_text += comment + '\n'
    return merged_text


def llm_call(messages: list[Message]) -> str:
    client = openai.OpenAI(api_key=os.environ['TOGETHER_API_KEY'], base_url='https://api.together.xyz/v1')
    res = client.chat.completions.create(
        model='meta-llama/Meta-Llama-3.1-70b-Instruct-Turbo',
        messages=[x.__dict__ for x in messages],
        temperature=0.8,
        max_tokens=1024
    )
    return res.choices[0].message.content
async def llm_call_async(messages: list[Message]) -> str: return await asyncio.to_thread(llm_call, messages)

async def get_pain_n_anger(text: str) -> list[str]:
    system_prompt = '''You are an expert at analyzing reddit posts and comment threads. Read through the provided post and comment and return the top 3 pain and anger points described by the users in the post. First describe the post and comment thread and then list out the pain and anger points inside triple backticks like below:
        ```pain_and_anger_points
        - <point 1>
        - <point 2>
        - <point 3>
        ```
    if there aren't any points, just mention "None"
    '''
    user_message = Message(role='user', content=text)
    system_message = Message(role='system', content=system_prompt)
    response = await llm_call_async([system_message, user_message])
    pain_n_anger_match = re.search(r'```pain_and_anger_points(.*?)```', response, re.DOTALL)
    pain_n_anger = pain_n_anger_match.group(1).strip().split('- ') if pain_n_anger_match else []
    pain_n_anger = [x.strip() for x in pain_n_anger if x and x != 'None']
    return pain_n_anger

async def get_advice_requests(text: str) -> list[str]:
    system_prompt = '''You are an expert at analyzing reddit posts and comment threads. Read through the provided post and comment and return the top 3 advice requests described by the users in the post. First describe the post and comment thread and then list out the advice requests inside triple backticks like below:
        ```advice_requests
        - <request 1>
        - <request 2>
        - <request 3>
        ```
    if there aren't any requests, just mention "None"
    '''
    user_message = Message(role='user', content=text)
    system_message = Message(role='system', content=system_prompt)
    response = await llm_call_async([system_message, user_message])
    advice_requests_match = re.search(r'```advice_requests(.*?)```', response, re.DOTALL)
    advice_requests = advice_requests_match.group(1).strip().split('\n- ') if advice_requests_match else []
    advice_requests = [x.strip() for x in advice_requests if x and x != 'None']
    return advice_requests

async def get_solution_requests(text: str) -> list[str]:
    system_prompt = '''You are an expert at analyzing reddit posts and comment threads. Read through the provided post and comment and return the top 3 solution requests described by the users in the post. First describe the post and comment thread and then list out the solution requests inside triple backticks like below:
        ```solution_requests
        - <request 1>
        - <request 2>
        - <request 3>
        ```
    if there aren't any requests, just mention "None"
    '''
    user_message = Message(role='user', content=text)
    system_message = Message(role='system', content=system_prompt)
    response = await llm_call_async([system_message, user_message])
    solution_requests_match = re.search(r'```solution_requests(.*?)```', response, re.DOTALL)
    solution_requests = solution_requests_match.group(1).strip().split('\n- ') if solution_requests_match else []
    solution_requests = [x.strip() for x in solution_requests if x and x != 'None']
    return solution_requests

async def generate_title(text: str) -> str:
    system_prompt = '''Below is a list of points that are part of a cluster. I want you to generate a single phrase that will encompass all the points. The phrase has to be 3-8 words maximum.
    You can take time to think out loud, before generating the final phrase, but you need to output the final phrase within triple backticks like below:
    ```
    <phrase>
    ```
    '''
    user_message = Message(role='user', content=text)
    system_message = Message(role='system', content=system_prompt)
    response = await llm_call_async([system_message, user_message])
    title = re.search(r'```(.*?)```', response, re.DOTALL)
    title = title.group(1).strip() if title else ''
    return title


EMBED_MODEL = SentenceTransformer('/Users/rohan/3_Resources/ai_models/all-mpnet-base-v2')
def embed(points: list[str]) -> list[list[float]]:
    """Use SentenceTransformer to embed each point"""
    embeddings = EMBED_MODEL.encode(points).tolist()
    return embeddings


def cluster(embeddings: list[list[float]], idx2post_id: dict[int, str], idx2point_id: dict[int, int], hottest_posts: list[Post], cluster_type: str) -> list[Cluster]:
    embeddings_array = np.array(embeddings)
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings_array)

    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(scaled_embeddings)
    labels = clustering.labels_
    
    clusters = []
    for label in set(labels):
        if label == -1: continue  # label = -1 meaning noise

        points_in_cluster = [idx2post_id[i] for i, x in enumerate(labels) if x == label]
        point_ids = [idx2point_id[i] for i, x in enumerate(labels) if x == label]
        cluster_posts = []
        for post_id, pid in zip(points_in_cluster, point_ids):
            post = next(post for post in hottest_posts if post.id == post_id)
            if cluster_type == 'pain_n_anger': point = post.pain_n_anger[idx2point_id[pid]]
            elif cluster_type == 'advice_requests': point = post.advice_requests[idx2point_id[pid]]
            elif cluster_type == 'solution_requests': point = post.solution_requests[idx2point_id[pid]]
            else: continue

            _tmp = ClusterPost(
                id=post.id,
                title=post.title,
                n_upvotes=post.n_upvotes,
                n_comments=post.n_comments,
                merged_content=post.merged_content,
                highlighted_point=point,
            )
            cluster_posts.append(_tmp)
        
        cluster = Cluster(
            id=str(label),
            posts=cluster_posts,
            n_upvotes=sum(post.n_upvotes for post in cluster_posts),
            n_comments=sum(post.n_comments for post in cluster_posts),
            n_posts=len(cluster_posts)
        )
        clusters.append(cluster)
    
    return clusters

def save_as_json(res, filename):
    data = []
    for cluster_type, clusters in zip(['pain_n_anger', 'advice_requests', 'solution_requests'], res):
        for cluster in clusters:
            cluster_data = cluster.to_dict()
            cluster_data['cluster_type'] = cluster_type
            data.append(cluster_data)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    subreddit_list = ['restaurant', 'restaurantowners', 'restaurateur', 'barowners']
    res = asyncio.run(main(subreddit_list))

    for r in res:
        for x in r: print(x.title)
        print('-'*20)

    # save res using pickle
    with open('cluster_results_2.pkl', 'wb') as f:
        pickle.dump(res, f)

    # save as JSON
    save_as_json(res, 'cluster_results_2.json')

