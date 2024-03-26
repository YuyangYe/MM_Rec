import openai
import os


os.environ['OPENAI_API_KEY'] = "sk-XLMKybMCQNuy4bJuB3Dc64C1AbA545F2893d91Bd23421dF2"
# os.environ['OPENAI_API_KEY'] = "sk-lEO0bBfYiaLypS7c85F5005731E84c18AaAf0e50A504901d" ## 这个令牌可以备用，我这边也显示正常

os.environ['OPENAI_API_BASE'] = "https://api.ai-yyds.com/v1"

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.environ['OPENAI_API_BASE']
openai.Model.list()


def query_chat_model(prompt):
    response = openai.ChatCompletion.create(
      model="gpt-4-vision-preview",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": prompt},
            {
              "type": "image_url",
              "image_url": {
                "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/3960.jpg",
              },
            },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/4022.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/6932.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/2914.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/7500.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/1.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/2.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/3.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/4.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/5.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/6.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/7.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/500.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/750.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/10.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/11.jpg",
                  },
              },              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/11.jpg",
                  },
              },
             {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/12.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/13.jpg",
                  },
              },

          ],
        }
      ],
      max_tokens=300,
    )
    reply = response['choices'][0]['message']['content']
    return reply,response


reply,response=query_chat_model("These are some video images which were interacted with a user in chronological. Can you summary the user's preference based on this image sequence?")
print(reply)
