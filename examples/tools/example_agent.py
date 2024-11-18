from griptape.structures import Agent
from griptape.plugin_name.tools.reverse_string import ReverseStringTool


agent = Agent(tools=[ReverseStringTool()])

agent.run("Use the ReverseStringTool to reverse 'Griptape'")