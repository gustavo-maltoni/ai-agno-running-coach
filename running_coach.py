import os
from dotenv import load_dotenv
import requests
from huami_token.zepp import ZeppSession
from datetime import datetime
from agno.agent import Agent
import litellm
from agno.models.litellm import LiteLLM
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()
litellm.drop_params = True

def get_physical_activity_history_from_zepp_tool():
    """Retrieve all physical activity history from Zepp Web API as a JSON"""

    try:
        session = ZeppSession(username=os.getenv('ZEPP_USERNAME'), password=os.getenv('ZEPP_PASSWORD'))
        session.login()

        headers = {
            "apptoken": session.app_token,
            "appname": "com.huami.midong",
            "user-agent": "Zepp/9.12.5",
        }

        #proxies = {
        #    'https': os.getenv('HTTPS_PROXY')
        #}

        params = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "apptoken": session.app_token
        }

        response = requests.get(
            '/v1/sport/data/summary.json',
            headers=headers,
            #proxies=proxies,
            params=params,
            timeout=5
        )

        return response.json()

    except:
        try:
            file = open('physical_history_example.json', 'r', encoding='utf-8')
            data = json.load(file)
            file.close()
            return data

        except Exception as e:
            return f'Exception: {e}'
    
def filter_running_history_tool(history_json):
    """Given the physical historical data as a JSON, filter only the running activities and related metrics"""
    summary = history_json['data']['summary']
    df = pd.DataFrame(summary)
    df = df[:9]

    df['trackid'] = df['trackid'].astype(int)
    df['calorie'] = df['calorie'].astype(float)
    df['date'] = pd.to_datetime(df['trackid'], unit='s')
    sns.lineplot(data=df, y='calorie', x='date')
    plt.show()

# Set up Agno agent
agent = Agent(
    system_message='You are a sports coach specialised in developing amateur runners to achieve high performance.',
    model=LiteLLM(
        id="gpt-5-mini",
        name="LiteLLM",
        api_base=os.getenv('LITELLM_BASE_URL'),
        temperature=0.7
    ),
    tools=[
        get_physical_activity_history_from_zepp_tool,
        filter_running_history_tool
    ],
    markdown=True
)

if __name__ == '__main__':
    response = get_physical_activity_history_from_zepp_tool()
    print(filter_running_history_tool(response))
    #agent.run('Retrieve my physical history data from Zepp, analyse my running stats, and build training for this week. My final goal is to accomplish a full marathon. Identify exclusive points of improvement by deep diving into my historical data.')