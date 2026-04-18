import os, requests, json
from dotenv import load_dotenv
from huami_token.zepp import ZeppSession
from datetime import datetime
from agno.agent import Agent
import litellm
from agno.models.litellm import LiteLLM
import pandas as pd

load_dotenv()
litellm.drop_params = True
HISTORY_FILENAME = 'history.csv'
ZEPP_BASE_URL = 'https://api-mifit-us2.zepp.com'
MONTHS_OFFSET = 6

def get_history_from_file():
    """Retrieve the physical activity history from a stored CSV file"""
    try:
        df = pd.read_csv(HISTORY_FILENAME)
        print(f"Loaded {len(df)} records from file")
        return df.to_dict('records')

    except Exception as e:
        return f'Exception loading history from file: {str(e)}'

def get_history_from_zepp_tool():
    """Retrieve all physical activity history from Zepp web API and store it into a CSV file"""
    try:
        if os.getenv('ZEPP_ENABLED') != str(True):
            return get_history_from_file()

        session = ZeppSession(username=os.getenv('ZEPP_USERNAME'), password=os.getenv('ZEPP_PASSWORD'))
        session.login()

        headers = {
            'apptoken': session.app_token,
            'appname': 'com.huami.midong',
            'user-agent': 'Zepp/9.12.5',
        }

        params = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'apptoken': session.app_token
        }

        url = f'{ZEPP_BASE_URL}/v1/sport/run/history.json'
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=5
        )

        history_json = response.json()

        df = pd.DataFrame(history_json['data']['summary'])
        df['date'] = pd.to_datetime(df['trackid'].astype(int), unit='s')
        date_limit = pd.Timestamp.now() - pd.DateOffset(months=MONTHS_OFFSET)

        summary_df = df[df['date'] >= date_limit]
        summary_df.to_csv(HISTORY_FILENAME, index=False)

        print(f"Retrieved {len(summary_df)} records from Zepp API")
        return summary_df.to_dict('records')

    except Exception as e:
        print(f"Error fetching from Zepp: {str(e)}")
        return get_history_from_file()
    
def get_running_categories_only_tool():
    """Provide a list of the running categories only to be filtered for an appropriate data manipulation"""
    return {
        '1': 'Outdoor running', # Eg: Track ID 1767731325
        '6': 'Walking', # Eg: Track ID 1767904310
        '8': 'Treadmill' # Eg: Track ID 1776107153
    }

def get_running_metrics_description_tool():
    """Provide a list of running-related metrics and their complementary description for an appropriate data manipulation"""
    return {
        'trackid': 'Activity ID and also its starting timestamp.',
        'dis': 'Running total distance in kilometres.',
        'calorie': 'Amount of calories burned during activity.',
        'end_time': 'Activity end timestamp.',
        'avg_pace': 'Average running pace calculated as total seconds per kilometre divided by 1000.',
        'avg_heart_rate': 'Average heart rate in bpm (beats per minute) during activity.',
        'type': 'Category type of the activity.',
        'city': 'City or neighbourhood where the outdoor activity took place.',
        'max_pace': 'Maximum running pace reached within the activity calculated as total seconds per kilometre divided by 1000.',
        'total_step': 'Total number of steps done during activity.',
        'max_heart_rate': 'Maximum heart rate in bpm (beats per minute) during activity.',
        'te': 'Coefficient of effect of aerobic training activity multiplied by 10.',
        'anaerobic_te': 'Coefficient of effect of anaerobic training activity multiplied by 10.',
        'weight': 'Current weight in kilos.'
    }

# Set up Agno agent
agent = Agent(
    system_message='You are a running coach specialised in developing amateur runners to achieve their high performance, helping them completing a full marathon.',
    model=LiteLLM(
        id="gpt-4o-mini",
        name="LiteLLM",
        api_base=os.getenv('LITELLM_BASE_URL'),
        temperature=0.7
    ),
    tools=[
        get_history_from_zepp_tool,
        get_running_categories_only_tool,
        get_running_metrics_description_tool
    ],
    markdown=True,
    debug_mode=True
)

if __name__ == '__main__':
    instructions = """
        1 - Retrieve the full history of my physical activity directly from Zepp.
        2 - Filter only the activities which belong to any running category type. The category types and their meaning can be retrieved by the get_running_categories_only_tool function.
            Example 2.1: If an activity has a type of "1", it means it is an outdoor running, so it should be considered.
            Example 2.2: If an activity has a type of "15", then it is not a running-related one. It can be ignored.
        3 - With the correct activities in hand, analyse their stats. Filter only the running-related metrics, which can be done through the get_running_metrics_description_tool function. The other ones can be ignored.
            Example 3.1: "max_pace" is a running-related metric and it indicates the maximum running pace reached within the activity. It is calculated as total seconds per kilometre divided by 1000, so it should be considered.
            Example 3.2: "climb_dis_descend" is not listed as a relevant running metric. It can be skipped.
        4 - Deep dive into my historical data and:
            4.1 - Highlight identified progress.
            4.2 - Identify exclusive points of improvement and build a succinct training plan for this week to overcome them."""

    agent.run(instructions)