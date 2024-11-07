from swarm import Swarm, Agent
from duckduckgo_search import DDGS
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

Model = "llama3.1"

#initialize swarm client

client = Swarm()
ddgs = DDGS()

#Searching on the web based on an given query

def search_web(query):
    print(f"Searching the web for {query} ...")

    #DuckDuckgo search
    current_date = datetime.now().strftime("%Y-%m")
    results = ddgs.text(f"{query} {current_date}", max_results=5)
    if results:
        new_results = ""
        for result in results:
            new_results += f"Tittle: {result['title']}\nURL: {result['href']}\nDescription: {result['body']}\n\n"
            return new_results.strip()
    else:
        return f"Could not find news results for {query}"


# Web search agent  to fetch lastest news

search_agent_web = Agent(
    name ="Web search Assistant",
    instructions=" Your role is to gather lastest news articles on specific topics using DuckDuckGo's search capabilities",
    functions=[search_web],
    model=Model
)

# Senior Reseach Analyst agent
researcher_agent = Agent(
    name = "Researcher agent",
    instructions= """ Your role is to analyze and synthesize the raw search results. You should:
    1. Remove duplicate information and redundant content.
    2. Identify and merge related topics and themes.
    3. Verify information consistency across sources
    4.Prioritize recent and relevent information
    5. Extract key facts, statistics, and quotes.
    6. Identify primary sources when available
    7. Flag any contradictory information
    8. Maintain proper attribution for important claims.
    9. Organize information in a logical sequence.
    10. Preserve important context and relationships between topics.
    """,
    model= Model

)

# editor agent , purpose to edit news
Writer_agent = Agent(
    name =" writer agent",
    instructions= """Your role is to transform the deduplicated research results into a polished, pubnlication-ready article. You should follow these instructions.
    1. Organize content into clear, thematic sections.
    2. Write in a professional yet engaging tone, that is genuine informative.
    3. Ensure proper flow between topics
    4. Add relveant context where needed.
    5. Maintain factual accuracy while making complex topics accessible.
    6. include a brief summary at the beginning and a resume at the end.
    7. Format with clear headlines and subheadings.
    8. Prepare all key information from the source material
    """,
    model=Model
)

# Create a run workflow

def workflow(query):
    print(f"Running web search assistant workflow ... ")

    # Search web
    news_response = client.run(
        agent= search_agent_web,
        messages = [{"role":"user", "content": f"Search the web for {query}"}],
    )
    raw_news = news_response.messages[-1]["content"]

    # Analyze and synthesize the search results
    research_analysis_response = client.run(
        agent=researcher_agent,
        messages= [{"role":"user", "content": raw_news}]
    )
    deduplicated_news = research_analysis_response.messages[-1]["content"]

    # Edit and plublish the analyze results
    return client.run(
        agent = Writer_agent,
        messages=[{"role": "user", "content": deduplicated_news}],
        stream= True
    )