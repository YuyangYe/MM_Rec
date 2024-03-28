import csv
import requests
import time
from multiprocessing import Pool

# OPENAI_API_KEY = "sk-XLMKybMCQNuy4bJuB3Dc64C1AbA545F2893d91Bd23421dF2"
OPENAI_API_KEY = "sk-lEO0bBfYiaLypS7c85F5005731E84c18AaAf0e50A504901d"
OPENAI_API_BASE = "https://api.ai-yyds.com/v1"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

N = 50

title_dicts = [{} for _ in range(N)]

used_vid = set()
with open('../data/MicroLens-50k/MicroLens-50k_summaries.csv', 'r') as f_summary:
    reader = csv.reader(f_summary)
    for row in reader:
        used_vid.add(row[0])

print("used video num:", len(used_vid))

with open('../data/MicroLens-50k/MicroLens-50k_titles.csv', 'r') as f_title:
    reader = csv.reader(f_title)
    reader.__next__()
    i = 0
    for row in reader:
        if row[0] in used_vid:
            continue
        title_dicts[i%N][row[0]] = row[1]
        i += 1
    print("unused video num:", i)

def summarize(video_id:str, title_dict:dict[str,str]):
    url = f"{OPENAI_API_BASE}/chat/completions"
    image_url = f"https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/{video_id}.jpg"
    prompt = f'this is the cover of a video with the title of "{title_dict[video_id]}". Please analyze the content in the cover and summarize the information of the video with its title and cover.'
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "low"
                    },
                }
            ]
        }
    ]
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 300
    }

    # 调用OpenAI的API
    response = None
    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        reply = response.json()['choices'][0]['message']['content']
        return reply
    except Exception as e:
        pp = f"[ERROR] {e.__class__.__name__}: {e}; video_id: {video_id}; response: '{response.content.decode()}'"
        print(pp)
        return pp

def task(x):
    i, title_dict = x
    with open(f'../data/MicroLens-50k/summaries/{i:02d}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        k = 0
        start = time.time()
        for vid in title_dict:
            writer.writerow([vid, summarize(vid, title_dict).replace('\n', '\\n')])
            k += 1
            if k % 10 == 0:
                csvfile.flush()
                now = time.time()
                length = now - start
                print(f"TASK {i:02d}: average time per request is {length/k:.2f}")

with Pool(N) as p:
    p.map(task, enumerate(title_dicts))