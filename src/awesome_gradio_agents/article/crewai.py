import asyncio
import threading
from typing import List, Dict, Any
import queue
import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import gradio as gr

class ArticleCrew:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.final_result = None
        self.current_agent = None

    def initialize_agents(self, topic: str):
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        os.environ["OPENAI_API_KEY"] = self.api_key
        llm = ChatOpenAI(temperature=0.7, model="gpt-4")

        self.planner = Agent(
            role="Content Planner",
            goal=f"Plan engaging and factually accurate content on {topic}",
            backstory="Expert content planner with focus on creating engaging outlines",
            allow_delegation=False,
            verbose=True,
            llm=llm
        )

        self.writer = Agent(
            role="Content Writer",
            goal=f"Write insightful and factually accurate piece about {topic}",
            backstory="Expert content writer with focus on engaging articles",
            allow_delegation=False,
            verbose=True,
            llm=llm
        )

        self.editor = Agent(
            role="Editor",
            goal="Polish and refine the article",
            backstory="Expert editor with eye for detail and clarity",
            allow_delegation=False,
            verbose=True,
            llm=llm
        )

    def create_tasks(self, topic: str) -> List[Task]:
        planner_task = Task(
            description=f"""Create a detailed content plan for an article about {topic} by:
1. Prioritizing the latest trends, key players, and noteworthy news
2. Identifying the target audience, considering their interests and pain points
3. Developing a detailed content outline including introduction, key points, and call to action
4. Including SEO keywords and relevant data or sources""",
            expected_output="A comprehensive content plan with outline, keywords, and target audience analysis",
            agent=self.planner
        )

        writer_task = Task(
            description="""Based on the provided content plan:
1. Use the content plan to craft a compelling blog post
2. Incorporate SEO keywords naturally
3. Ensure sections/subtitles are properly named in an engaging manner
4. Create proper structure with introduction, body, and conclusion
5. Proofread for grammatical errors""",
            expected_output="A well-written article draft following the content plan",
            agent=self.writer
        )

        editor_task = Task(
            description="""Review the written article by:
1. Checking for clarity and coherence
2. Correcting any grammatical errors and typos
3. Ensuring consistent tone and style
4. Verifying proper formatting and structure""",
            expected_output="A polished, final version of the article ready for publication",
            agent=self.editor
        )

        return [planner_task, writer_task, editor_task]

async def stream_from_agent(prompt: str, chatbot: list[dict], api_key: str):
    article_crew = ArticleCrew(api_key=api_key)
    
    yield gr.Textbox(interactive=False), gr.skip(), gr.skip()
    chatbot.append({"role": "user", "content": f"Write an article about: {prompt}"})
    yield gr.skip(), chatbot, gr.skip()

    try:
        article_crew.initialize_agents(prompt)
        article_crew.current_agent = "Content Planner"

        # Start process
        chatbot.append({
            "role": "assistant",
            "content": "Starting work on your article...",
            "metadata": {"title": "🚀 Process Started"}
        })
        yield gr.skip(), chatbot, gr.skip()

        # Initialize first agent
        chatbot.append({
            "role": "assistant",
            "content": "Content Planner",
            "metadata": {"title": "🤖 Content Planner"}
        })
        chatbot.append({
            "role": "assistant",
            "content": """1. Prioritize the latest trends, key players, and noteworthy news
2. Identify the target audience, considering their interests and pain points
3. Develop a detailed content outline including introduction, key points, and call to action
4. Include SEO keywords and relevant data or sources""",
            "metadata": {"title": "📋 Task for Content Planner"}
        })
        yield gr.skip(), chatbot, gr.skip()

        final_result = None
        editor_result = None

        def task_callback(task_output):
            nonlocal final_result, editor_result
            print(f"Task callback received from {article_crew.current_agent}")
            
            raw_output = task_output.raw
            if "## Final Answer:" in raw_output:
                content = raw_output.split("## Final Answer:")[1].strip()
            else:
                content = raw_output.strip()
            
            if article_crew.current_agent == "Editor":
                editor_result = content
            else:
                # For other agents, show their output with metadata
                chatbot.append({
                    "role": "assistant",
                    "content": content,
                    "metadata": {"title": f"✨ Output from {article_crew.current_agent}"}
                })
                
                next_agent = {"Content Planner": "Content Writer", "Content Writer": "Editor"}.get(article_crew.current_agent)
                if next_agent:
                    article_crew.current_agent = next_agent
                    tasks = {
                        "Content Writer": """1. Use the content plan to craft a compelling blog post
2. Incorporate SEO keywords naturally
3. Ensure sections/subtitles are properly named in an engaging manner
4. Create proper structure with introduction, body, and conclusion
5. Proofread for grammatical errors""",
                        "Editor": """1. Review the article for clarity and coherence
2. Check for grammatical errors and typos
3. Ensure consistent tone and style
4. Verify proper formatting and structure"""
                    }
                    chatbot.append({
                        "role": "assistant",
                        "content": f"Moving on to {next_agent}",
                        "metadata": {"title": f"🤖 {next_agent}"}
                    })
                    chatbot.append({
                        "role": "assistant",
                        "content": tasks[next_agent],
                        "metadata": {"title": f"📋 Task for {next_agent}"}
                    })

        crew = Crew(
            agents=[article_crew.planner, article_crew.writer, article_crew.editor],
            tasks=article_crew.create_tasks(prompt),
            verbose=True,
            task_callback=task_callback
        )

        def run_crew():
            try:
                result = crew.kickoff()
                print("Crew execution completed")
                if editor_result:
                    # Simply pass through the final article without any modifications
                    chatbot.append({
                        "role": "assistant",
                        "content": "Final article is ready!",
                        "metadata": {"title": "📝 Final Article"}
                    })
                    
                    chatbot.append({
                        "role": "assistant",
                        "content": editor_result
                    })
            except Exception as e:
                print(f"Error in crew execution: {str(e)}")
                chatbot.append({
                    "role": "assistant",
                    "content": f"An error occurred: {str(e)}",
                    "metadata": {"title": "❌ Error"}
                })
                    
                chatbot.append({
                        "role": "assistant",
                        "content": "Final article is ready!",
                        "metadata": {"title": "📝 Final Article"}
                    })
                    
                chatbot.append({
                        "role": "assistant",
                        "content": formatted_content
                    })
            except Exception as e:
                print(f"Error in crew execution: {str(e)}")
                chatbot.append({
                    "role": "assistant",
                    "content": f"An error occurred: {str(e)}",
                    "metadata": {"title": "❌ Error"}
                })

        thread = threading.Thread(target=run_crew)
        thread.start()

        while thread.is_alive():
            yield gr.skip(), chatbot, gr.skip()
            await asyncio.sleep(0.1)

        # One final yield after thread completion
        yield gr.skip(), chatbot, gr.skip()

    except Exception as e:
        print(f"Error in stream_from_agent: {str(e)}")
        chatbot.append({
            "role": "assistant",
            "content": f"An error occurred: {str(e)}",
            "metadata": {"title": "❌ Error"}
        })
        yield gr.skip(), chatbot, gr.skip()

async def handle_retry(chatbot, past_messages: list, retry_data: gr.RetryData):
    new_history = chatbot[: retry_data.index]
    previous_prompt = chatbot[retry_data.index]["content"]
    past_messages = past_messages[: retry_data.index]
    async for update in stream_from_agent(previous_prompt, new_history, None):
        yield update

def undo(chatbot, past_messages: list, undo_data: gr.UndoData):
    new_history = chatbot[: undo_data.index]
    past_messages = past_messages[: undo_data.index]
    return chatbot[undo_data.index]["content"], new_history, past_messages

def select_data(message: gr.SelectData) -> str:
    return message.value["text"]