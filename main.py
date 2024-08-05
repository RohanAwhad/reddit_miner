import dataclasses
import numpy as np
import openai
import os
import praw

from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Optional

post_id = 0
@dataclasses.dataclass
class HotPostsOutput:
    id: int
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
    posts: list[str]
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
def get_hot_posts_and_comments(subreddit_name, limit) -> list[HotPostsOutput]:
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
        
        result.append(HotPostsOutput(
            id = post_id,
            title=post.title,
            selftext=post.selftext,
            n_upvotes=n_upvotes,
            n_comments=len(comment_list),
            comments=comment_list
        ))
        post_id += 1

    return result


llm = openai.OpenAI(api_key=os.environ['TOGETHER_API_KEY'], base_url='https://api.together.xyz/v1')
model = SentenceTransformer('/Users/rohan/3_Resources/ai_models/all-mpnet-base-v2')
cross_encoder = CrossEncoder('/Users/rohan/3_Resources/ai_models/ms-marco-MiniLM-L-6-v2')


def main(subreddit_list: list[str]):
    # get a subreddit list
    # for each subreddit get the top 100 hottest posts
    # for each of those post summarize the entire comment thread into a single paragraph
    # then merge each of the posts into a single long str with title, post and comments summary
    # generate embeddings for each of the long strs of the post
    # use DBSCAN clustering to cluster similar posts into a single one.
    # Summarize each cluster into multiple sections: Pain & Anger, Advice Requests and Solution Requests
    # Create title's for each cluster from Pain&Anger, Advice Request, Solution Request POV.
    # Use cross encoder to rerank title-cluster matching. And then get 10 (or 20) clusters for each POV, with the title. 

    hottest_posts: list[HotPostsOutput] = []
    print('Gathering posts ...')
    for sr in subreddit_list: hottest_posts.extend(get_hot_posts_and_comments(sr, limit=10))

    print('Summarizing posts and creating embeddings ...')
    for post in hottest_posts:
        post.summarized_comment_thread = summarize_comment_thread(post.comments) 
        post.merged_content = merge_post(post)
        post.embeddings = calculate_embeddings(post.merged_content)

    post_id_to_idx = {post.id: idx for idx, post in enumerate(hottest_posts)}
    all_embeddings = []
    for idx, post in enumerate(hottest_posts):
        post_id_to_idx[post.id] = idx
        all_embeddings.append(post.embeddings)

    all_embeddings = np.array(all_embeddings)
    print('Clustering posts ...')
    clusters: list[Cluster] = cluster_posts(all_embeddings, post_id_to_idx, hottest_posts)
    pain_n_anger_clusters = []
    advice_requests_clusters = []
    solution_requests_clusters = []

    print('Classifying clusters ...') 
    for cl in clusters:
        cl.summary = summarize_cluster(cl.posts)

        cl.pain_n_anger_title = get_pain_n_anger_title(cl.summary)
        cl.advice_requests_title = get_advice_requests_title(cl.summary)
        cl.solution_requests_title = get_solution_requests_title(cl.summary)

        cl.pain_n_anger_score = get_pain_n_anger_score(cl.pain_n_anger_title, cl.summary)
        cl.advice_requests_score = get_advice_requests_score(cl.advice_requests_title, cl.summary)
        cl.solution_requests_score = get_solution_requests_score(cl.solution_requests_title, cl.summary)

        # see which has the highest score and add to that list
        highest_score = max(cl.pain_n_anger_score, cl.advice_requests_score, cl.solution_requests_score)
        if highest_score == cl.pain_n_anger_score: pain_n_anger_clusters.append(cl)
        elif highest_score == cl.advice_requests_score: advice_requests_clusters.append(cl)
        else: solution_requests_clusters.append(cl)

    # Processed clusters are now categorized
    return {
        "pain_n_anger_clusters": pain_n_anger_clusters,
        "advice_requests_clusters": advice_requests_clusters,
        "solution_requests_clusters": solution_requests_clusters
    }

def summarize_comment_thread(comments_list: list[str]) -> str:
    comments_text = " ".join(comments_list)
    response = llm.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{'role': 'user', 'content': f"Summarize the following comment thread: {comments_text}\n\n---\n\nDO NOT OUTPUT ANYTHING ELSE. ONLY SUMMARY."}],
        max_tokens=4096,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def merge_post(post: HotPostsOutput) -> str:
    return f"Title: {post.title}\n\nPost: {post.selftext}\n\nComments Summary: {post.summarized_comment_thread}"

def calculate_embeddings(text: str) -> list[float]:
    return model.encode(text).tolist()

def cluster_posts(embeddings: np.ndarray, post_id_to_idx: dict, posts: list[HotPostsOutput]) -> list[Cluster]:
    clustering_model = DBSCAN(eps=0.5, min_samples=5, metric='cosine').fit(embeddings)
    cluster_labels = clustering_model.labels_
    
    clusters_dict = {}
    for post_idx, cluster_id in enumerate(cluster_labels):
        if cluster_id == -1: continue
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = Cluster(id=cluster_id, posts=[], total_posts=0, total_upvotes=0, total_comments=0)
        clusters_dict[cluster_id].posts.append(posts[post_idx].merged_content)
        clusters_dict[cluster_id].total_posts += 1
        clusters_dict[cluster_id].total_upvotes += posts[post_idx].n_upvotes
        clusters_dict[cluster_id].total_comments += posts[post_idx].n_comments
    
    return list(clusters_dict.values())

def summarize_cluster(post_contents: list[str]) -> str:
    combined_contents = " ".join(post_contents)
    response = llm.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{'role': 'user', 'content': f"Summarize the following contents: {combined_contents}\n\n---\n\nDO NOT OUTPUT ANYTHING ELSE. ONLY SUMMARY."}],
        max_tokens=4096,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def get_pain_n_anger_title(summary: str) -> str:
    response = llm.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{'role': 'user', 'content': f"Generate a title focused on Pain & Anger from the following summary: {summary}"}],
        max_tokens=100,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def get_advice_requests_title(summary: str) -> str:
    response = llm.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{'role': 'user', 'content': f"Generate a title focused on Advice Requests from the following summary: {summary}"}],
        max_tokens=100,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def get_solution_requests_title(summary: str) -> str:
    response = llm.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{'role': 'user', 'content': f"Generate a title focused on Solution Requests from the following summary: {summary}"}],
        max_tokens=100,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def get_pain_n_anger_score(title: str, summary: str) -> float:
    return cross_encoder.predict([(title, summary)])[0]

def get_advice_requests_score(title: str, summary: str) -> float:
    return cross_encoder.predict([(title, summary)])[0]

def get_solution_requests_score(title: str, summary: str) -> float:
    return cross_encoder.predict([(title, summary)])[0]



if __name__ == '__main__':
    subreddit_list = ['restaurant', 'restaurantowners', 'restaurateur', 'barowners']
    result = main(subreddit_list)
    print(result)
