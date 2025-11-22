import os
from openai import OpenAI


# Define the function to interact with the model and check the response
def check_answer(query, correct_answer, model_prediction, llm_config):
    prompt = f"""
    Please determine wether the model's prediction is correct or not, the model's prediction may have the different format from the actual answer. Here are some examples:
    - Query: “What is 1/2 + 1/3?”
	- Correct Answer: “5/6”
	- Model Prediction: “0.8333”
	- Explanation: The model gives the decimal form of the answer while the correct answer is the fraction form. These are mathematically equivalent answers.
	- Your Response: True

    - Query: “What is the capital of Japan?”
	- Correct Answer: “Tokyo”
	- Model Prediction: “The capital of Japan is Tokyo.”
	- Explanation: The model’s prediction is a full sentence answer, but the key information (“Tokyo”) is still correct and included within the sentence.
	- Your Response: True
    
    - Query: “What is the speed of light?”
	- Correct Answer: “299,792,458 meters per second”
	- Model Prediction: “3 × 10^8 meters per second”
	- Explanation: The model represents the speed of light using scientific notation, which is mathematically equivalent to the exact number.
	- Your Response: True

    - Query: “Who was the first president of the United States?”
	- Correct Answer: “George Washington”
	- Model Prediction: “Thomas Jefferson”
	- Explanation: Thomas Jefferson was the third president, not the first. The model incorrectly identifies him instead of George Washington.
	- Result: False

    Given the following information:
    
    """
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=llm_config["config_list"][0]["api_key"],
        base_url=llm_config["config_list"][0]["base_url"],
    )

    query = "Query: " + query + "\n"
    correct_answer = "Correct Answer: " + correct_answer + "\n"
    model_prediction = "Model's Prediction: " + model_prediction + "\n"
    others = """
    Please determine if the model's prediction is correct. 
    Do not explain the reason and only return "True" or "False".
    """
    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                        + query
                        + correct_answer
                        + model_prediction
                        + others,
                    },
                ],
            }
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    ans = completion.choices[0].message.content.strip().lower()
    return ans


if __name__ == "__main__":
    # Example usage:
    query = "What is the capital of France?"
    correct_answer = "Paris"
    model_prediction = "London"

    # Output will be False in this case
    print(check_answer(query, correct_answer, model_prediction))
