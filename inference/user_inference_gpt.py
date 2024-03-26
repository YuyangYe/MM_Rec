import csv
import openai
import os


os.environ['OPENAI_API_KEY'] = "sk-XLMKybMCQNuy4bJuB3Dc64C1AbA545F2893d91Bd23421dF2"
# os.environ['OPENAI_API_KEY'] = "sk-lEO0bBfYiaLypS7c85F5005731E84c18AaAf0e50A504901d" ## 这个令牌可以备用，我这边也显示正常

os.environ['OPENAI_API_BASE'] = "https://api.ai-yyds.com/v1"

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.environ['OPENAI_API_BASE']
openai.Model.list()


def query_chat_model(prompt, item_list):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    for item in item_list:
        image_url = f"https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/{item}.jpg"
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    # 调用OpenAI的API
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=300,
    )

    reply = response['choices'][0]['message']['content']
    return reply, response


user_history = {}
f = open('../data/MicroLens-50k/MicroLens-50k_pairs.tsv', 'r')
for line in f:
    user, items = line.strip().split('\t')
    user_history[user] = items


title_dict = {}
with open('../data/MicroLens-50k/MicroLens-50k_titles.csv', 'r') as f_title:
    reader = csv.reader(f_title)
    for row in reader:
        title_dict[row[0]] = row[0] + ':' + row[1]

i = 0
for k, v in user_history.items():
    item_list = v.split()
    title_str = ' '.join([title_dict[item] for item in item_list])
    reply,response=query_chat_model("These are some video images which were interacted with a user in chronological order. The video titles are as follows:" + title_str + "Can you summary the user's preference based on this image sequence??", item_list)
    print(reply)

