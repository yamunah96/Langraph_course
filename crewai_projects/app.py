from crewai import Agent, Task,Crew, LLM
from crewai_tools import SerperDevTool

from dotenv import load_dotenv
load_dotenv()

topic="Medical Industry using Generative AI"
import os

# crewai's native tool-calling loop tags every message with a synthetic
# "cache_breakpoint" key for providers that need it, but never strips it
# for models routed through the generic LiteLLM path (e.g. Groq). Groq's
# strict schema validator then 400s on the unknown property, so disable
# the marker entirely.
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda message: message

# tool1
llm=LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=os.getenv('GROQ_API_KEY'),
        temperature=0.2,
        max_tokens=2000
)

# tool2
search_tool= SerperDevTool(n_results=10)

# agent 1
senior_research_analyst=Agent(
    role='Senior Research Analyst',
    goal =f"Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources",
    backstory="You're an expert research analyst with advance web research skilss, You excel at finding, analyzing, "
    "and synthesizing information from,across the internet using search tools, youre skilled at distingushing reliable sources from unreliable ones",
    allow_delgation=False,
    verbose=True,
    tools=[search_tool],
    llm=llm 
)

# agent 2 content writer
content_writer= Agent(
     role='Content Writer',
    goal =f"Transform research findings into clear, engaging, and well-structured content on {topic} for a general audience",
    backstory="You're a skilled content writer with expertise in transforming complex research findings into clear, engaging, "
    "and well-structured content for a general audience. You excel at synthesizing information and presenting it in a way that is accessible and compelling.",
   allow_delegation=False,
    verbose=True,
    llm=llm 
)


# Task 1 Research Tasks
research_task= Task(
    description=("""Conduct in-depth research on {topic} using reliable web sources, analyze the information, and synthesize it into a comprehensive 
                 reportevaluate the credibility of sources, extract key insights, and organize the findings in a structured format. organize the research findings into a clear 
                 and concise report that highlights the most relevant information, key trends, and actionable insights related to {topic}. The report should be well-structured, easy to 
                 understand, and suitable for presentation to stakeholders or decision-makers.
    Include all relevant data, statistics, and references to support the findings. The final report should be comprehensive, accurate, and provide a clear understanding of the current state of {topic} in the medical industry.",
    """),
    expected_output=""" A detailed research report containing
            - Executive summary of key findings
            - All citations and links to original sources
            - list of verified facts and statistics""",
    agent=senior_research_analyst
)
#Task 2 content Tasks
content_writing_task=Task(
    description="""
     using the resaerch brief provided , create an engaging blog post that:
        1.Transforms techinal information into accessible content
    """,
    expected_output="""
        -A polsied blog post in markdown format that conatins reader engagenmetm, structed section, includes inline citaions
    """,
    agent=content_writer
)


# crew
crew= Crew(
    agents=[senior_research_analyst,content_writer],
    tasks=[research_task,content_writing_task],
    verbose=True
)

result= crew.kickoff(inputs={"topic":topic})
print(result)