from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.tools import PythonREPLTool

load_dotenv()


def main():
    print("Start")
    
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4o", temperature=0),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True
    )
    
    csv_agent.invoke({"input": "How many episodes each season has?"})


if __name__ == "__main__":
    main()
