import asyncio
import dataclasses
import numpy as np
import openai
import os
import praw

from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Optional, Any

@dataclasses.dataclass
class Post:
    id: str
    title: str
    selftext: str
    n_upvotes: int
    n_comments: int
    comments: list[str]
    summarized_comment_thread: str = ''
    merged_content: str = ''
    cluster_id: int = -1
    embeddings: Optional[list[float]] = None


@dataclasses.dataclass
class Cluster:
    id: int 
    posts: list[Post]
    total_posts: int = 0
    total_upvotes: int = 0
    total_comments: int = 0
    summary: str = ''
    pain_n_anger_title: str = ''
    advice_requests_title: str = ''
    solution_requests_title: str = ''
    pain_n_anger_score: float = 0.0
    advice_requests_score: float = 0.0
    solution_requests_score: float = 0.0



# write a function that will take in a subreddit name and put a list of top 100
# posts in that subreddit. Then also for each post, get all the comments and the reply thread
# and finally return a list[HotPostsOutput], where the nested list contains submission post and its
# comments. Use praw library
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


llm = openai.OpenAI(api_key=os.environ['TOGETHER_API_KEY'], base_url='https://api.together.xyz/v1')
model = SentenceTransformer('/Users/rohan/3_Resources/ai_models/all-mpnet-base-v2')
cross_encoder = CrossEncoder('/Users/rohan/3_Resources/ai_models/ms-marco-MiniLM-L-6-v2')


BATCH_SIZE = 50
async def main(subreddit_list: list[str]):
    # get a subreddit list
    # for each subreddit get the top 100 hottest posts
    # for each of those post summarize the entire comment thread into a single paragraph
    # then merge each of the posts into a single long str with title, post and comments summary
    # generate embeddings for each of the long strs of the post
    # use DBSCAN clustering to cluster similar posts into a single one.
    # Summarize each cluster into multiple sections: Pain & Anger, Advice Requests and Solution Requests
    # Create title's for each cluster from Pain&Anger, Advice Request, Solution Request POV.
    # Use cross encoder to rerank title-cluster matching. And then get 10 (or 20) clusters for each POV, with the title. 

    hottest_posts: list[Post] = []
    print('Gathering posts ...')
    for sr in subreddit_list: hottest_posts.extend(get_hot_posts_and_comments(sr, limit=20))

    print(hottest_posts)

    print('Summarizing posts ...')
    for i in range(0, len(hottest_posts), BATCH_SIZE):
        coros = [summarize_comment_thread_async(x.comments) for x in hottest_posts[i:i+BATCH_SIZE]]
        result = await asyncio.gather(*coros)
        for x, y in zip(hottest_posts[i:i+BATCH_SIZE], result):
            x.summarized_comment_thread = y

    print(hottest_posts[-1].summarized_comment_thread)
    print('Creating embeddings ...')
    for post in hottest_posts:
        post.merged_content = merge_post(post)
        post.embeddings = calculate_embeddings(post.merged_content)

    all_embeddings = []
    for post in hottest_posts: all_embeddings.append(post.embeddings)

    all_embeddings = np.array(all_embeddings)
    print('Clustering posts ...')
    clusters: list[Cluster] = cluster_posts(all_embeddings, hottest_posts)
    print(clusters)

    print('Generating titles ...') 
    for i in range(0, len(clusters), BATCH_SIZE):
        await asyncio.gather(*[generate_titles(x) for x in clusters[i:i+BATCH_SIZE]])

    for cl in clusters:
        cl.pain_n_anger_score = cross_encoder.predict([(cl.pain_n_anger_title, cl.summary)])[0]
        cl.advice_requests_score = cross_encoder.predict([(cl.advice_requests_title, cl.summary)])[0]
        cl.solution_requests_score = cross_encoder.predict([(cl.solution_requests_title, cl.summary)])[0]


    # Processed clusters are now categorized
    return {
        "pain_n_anger_clusters": sorted(clusters, key=lambda x: x.pain_n_anger_score, reverse=True),
        "advice_requests_clusters": sorted(clusters, key=lambda x: x.advice_requests_score, reverse=True),
        "solution_requests_clusters": sorted(clusters, key=lambda x: x.solution_requests_score, reverse=True)
    }

async def generate_titles(cluster: Cluster):
    cluster.summary = await summarize_cluster_async(cluster.posts)
    titles = await asyncio.gather(*[
        get_pain_n_anger_title_async(cluster.summary),
        get_advice_requests_title_async(cluster.summary),
        get_solution_requests_title_async(cluster.summary),
    ])
    cluster.pain_n_anger_title = titles[0]
    cluster.advice_requests_title = titles[1]
    cluster.solution_requests_title = titles[2]


def summarize_comment_thread(comments_list: list[str]) -> str:
    comments_text = "\n\n---\n\n".join(comments_list)
    response = llm.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{'role': 'user', 'content': f"Summarize the following comment thread:\n'''\n{comments_text}\n'''\n\nDO NOT OUTPUT ANYTHING ELSE. ONLY SUMMARY."}],
        max_tokens=4096,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()
async def summarize_comment_thread_async(comments_list: list[str]) -> str: return await asyncio.to_thread(summarize_comment_thread, comments_list)

def merge_post(post: Post) -> str:
    return f"Title: {post.title}\n\nPost:\n{post.selftext}\n\nComments Summary:\n{post.summarized_comment_thread}"

def calculate_embeddings(text: str) -> list[float]:
    return model.encode(text).tolist()

def cluster_posts(embeddings: np.ndarray, posts: list[Post]) -> list[Cluster]:
    clustering_model = DBSCAN(eps=0.5, min_samples=5, metric='cosine').fit(embeddings)
    cluster_labels = clustering_model.labels_
    
    clusters_dict = {}
    for post_idx, cluster_id in enumerate(cluster_labels):
        if cluster_id == -1: continue
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = Cluster(id=cluster_id, posts=[], total_posts=0, total_upvotes=0, total_comments=0)
        clusters_dict[cluster_id].posts.append(posts[post_idx])
        clusters_dict[cluster_id].total_posts += 1
        clusters_dict[cluster_id].total_upvotes += posts[post_idx].n_upvotes
        clusters_dict[cluster_id].total_comments += posts[post_idx].n_comments
    
    return list(clusters_dict.values())

def summarize_cluster(post_contents: list[Post]) -> str:
    combined_contents = "\n---\n".join([x.merged_content for x in post_contents])
    response = llm.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{'role': 'user', 'content': f"Summarize the following contents:\n'''\n{combined_contents}\n'''\n\nDO NOT OUTPUT ANYTHING ELSE. ONLY SUMMARY."}],
        max_tokens=4096,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()
async def summarize_cluster_async(post_contents: list[Post]) -> str: return await asyncio.to_thread(summarize_cluster, post_contents)

def get_pain_n_anger_title(summary: str) -> str:
    response = llm.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{'role': 'user', 'content': f"Summarize the Pain and Anger mentioned in the below text in ONE single line:\n\n{summary}"}],
        max_tokens=100,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()
async def get_pain_n_anger_title_async(summary: str) -> str: return await asyncio.to_thread(get_pain_n_anger_title, summary)

def get_advice_requests_title(summary: str) -> str:
    response = llm.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{'role': 'user', 'content': f"Summarize the Advice requests mentioned in the below text in ONE single line:\n\n{summary}"}],
        max_tokens=100,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()
async def get_advice_requests_title_async(summary: str) -> str: return await asyncio.to_thread(get_advice_requests_title, summary)

def get_solution_requests_title(summary: str) -> str:
    response = llm.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{'role': 'user', 'content': f"Summarize the Solution requests mentioned in the below text in ONE single line:\n\n{summary}"}],
        max_tokens=100,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()
async def get_solution_requests_title_async(summary: str) -> str: return await asyncio.to_thread(get_solution_requests_title, summary)



@dataclasses.dataclass
class ResultPost:
    id: str
    title: str
    body: str
    comments_summary: str
    post_summary: str
    n_upvotes: int
    n_comments: int

@dataclasses.dataclass
class ResultCluster:
    id: int
    pain_n_anger_title: str
    advice_requests_title: str
    solution_requests_title: str
    pain_n_anger_score: float
    advice_requests_score: float
    solution_requests_score: float
    posts: list[dict[str, Any]]  # list[ResultPost]
    cluster_summary: str


if __name__ == '__main__':
    subreddit_list = ['restaurant', 'restaurantowners', 'restaurateur', 'barowners']
    result = asyncio.run(main(subreddit_list))

    # modify result into the respective class list
    pain_n_anger_clusters = [
        ResultCluster(
            id=cl.id,
            pain_n_anger_title=cl.pain_n_anger_title,
            advice_requests_title=cl.advice_requests_title,
            solution_requests_title=cl.solution_requests_title,
            pain_n_anger_score=float(cl.pain_n_anger_score),
            advice_requests_score=float(cl.advice_requests_score),
            solution_requests_score=float(cl.solution_requests_score),
            posts=[
                ResultPost(
                    id=post.id,
                    title=post.title,
                    body=post.selftext,
                    comments_summary=post.summarized_comment_thread,
                    post_summary=post.merged_content,
                    n_upvotes=post.n_upvotes,
                    n_comments=post.n_comments
                ).__dict__ for post in cl.posts
            ],
            cluster_summary=cl.summary
        ) for cl in result['pain_n_anger_clusters']
    ]

    advice_requests_clusters = [
        ResultCluster(
            id=cl.id,
            pain_n_anger_title=cl.pain_n_anger_title,
            advice_requests_title=cl.advice_requests_title,
            solution_requests_title=cl.solution_requests_title,
            pain_n_anger_score=float(cl.pain_n_anger_score),
            advice_requests_score=float(cl.advice_requests_score),
            solution_requests_score=float(cl.solution_requests_score),
            posts=[
                ResultPost(
                    id=post.id,
                    title=post.title,
                    body=post.selftext,
                    comments_summary=post.summarized_comment_thread,
                    post_summary=post.merged_content,
                    n_upvotes=post.n_upvotes,
                    n_comments=post.n_comments
                ).__dict__ for post in cl.posts
            ],
            cluster_summary=cl.summary
        ) for cl in result['advice_requests_clusters']
    ]

    solution_requests_clusters = [
        ResultCluster(
            id=cl.id,
            pain_n_anger_title=cl.pain_n_anger_title,
            advice_requests_title=cl.advice_requests_title,
            solution_requests_title=cl.solution_requests_title,
            pain_n_anger_score=float(cl.pain_n_anger_score),
            advice_requests_score=float(cl.advice_requests_score),
            solution_requests_score=float(cl.solution_requests_score),
            posts=[
                ResultPost(
                    id=post.id,
                    title=post.title,
                    body=post.selftext,
                    comments_summary=post.summarized_comment_thread,
                    post_summary=post.merged_content,
                    n_upvotes=post.n_upvotes,
                    n_comments=post.n_comments
                ).__dict__ for post in cl.posts
            ],
            cluster_summary=cl.summary
        ) for cl in result['solution_requests_clusters']
    ]

    # Output the results
    final_result = {
        "pain_n_anger_clusters": [x.__dict__ for x in pain_n_anger_clusters],
        "advice_requests_clusters": [x.__dict__ for x in advice_requests_clusters],
        "solution_requests_clusters": [x.__dict__ for x in solution_requests_clusters]
    }
    import pickle
    with open('final_results.pkl', 'wb') as f: pickle.dump(final_result, f)

    import json
    print(json.dumps(final_result, indent=2))
    
    # save the final result in a json with indent=2

    with open('final_result.json', 'w') as f:
        json.dump(final_result, f, indent=2)

    print("Results saved to 'final_result.json'.")




