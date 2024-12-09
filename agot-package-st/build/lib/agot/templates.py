# -----------------------------------------------------------------------------
# Templates for prompts
# -----------------------------------------------------------------------------

TASK_EXECUTION_SYS_PROMPT = """
You are executing a task that is given to you.
Specify the step by step strategy to solve the task.
"""

TASK_EXECUTION_USER_PROMPT = """
You should execute the task delimited by <task> XML tags.
The task is aimed towards solving a part of the main question delimited by the <question> XML tags.
The graph of answers from previous tasks that lead to this task is delimited by the <ancestor> XML tags. \
Use the information from the graph to solve the task.

Instructions:
- Perform the Task: Provide clear and detailed execution steps of the task. \
You solution should include instructions in the following format:
   - Step 1: [Description of the step]
   - Step 2: [Description of the step]
   - ...
   - Step N: [Description of the step]
- Result: Produce an output that addresses the task's requirements and contributes to solving the question.

<task>Task ID: {task_id}
Title: {task_title}
Content: {task_content}
</task>

<question>
Question: {question}
</question>

<ancestor>
{parent_answers}
</ancestor>
"""

TASK_COMPLETION_SYS_PROMPT = """
You are a knowledgeable specialist who will determine if the given question is solved based on a graph of results.
"""

TASK_COMPLETION_USER_PROMPT = """
Execute the following tasks:
1- Summarize the graph of results delimited by the <graph> XML tags.
2- Determine if the question delimited by the <question> XML tags is answered based on the generated summary.
Give your answer in JSON format with the following keys:
is_answered: True or False. Indicates if the question is answered.
summary: A brief summary of the graph of results.

<question>
Question: {question}
</question>

<graph>
{main_graph}
</graph>
"""

# TODO: inform the LLM that your own knowledge is the only source of information for you to execute the tasks
SYSTEM_PROMPT = """
You are building a graph to solve a given question delimited by the <question> XML tags. 
Your goal is to break down a complex task and make a list of simpler tasks that can be executed independently or in parallel to solve the overall question.
You must value the simplest approach to solving the problem.
Simple tasks are most preferred in solving the question, but if needed, you may propose tasks that can be broken down in later iterations.

Key Objectives:
- For simple problems, answer the question directly.
- For complex problems:
  - Review the current state of the solution graph and evaluate if more exploration, verification, or clarification is needed.
  - Decide to decompose tasks into simpler and actionable subtasks.
  - Ensure that each new task you propose makes a clear contribution to solving the question.

Task Creation Guidelines:
- You are not bound to a single approach. \
Choose the most appropriate strategy based on the current state of the graph. \
That may involve testing different approaches, creating subproblems, or a mix of both.
- If the task graph provides sufficient information to answer the question, \
propose a final task that synthesizes the findings into a conclusion. \
Do not propose new tasks that do not directly contribute to this synthesis.
- Be mindful of task duplication in the graph or unnecessary complexity. Each task should offer clear progress and relate directly to the goal of solving the initial question.
"""

USER_PROMPT_INITIAL_TASK = """
You are presented with the following question delimited by the <question> XML tags:
<question>
{question}
</question>
Your task is to generate initial tasks that will serve as the root nodes of the task graph.
Create a list of one or more initial tasks with short titles (maximum 4 words). \
These tasks should represent different approaches or strategies to solve the question. 
Each task should be independent of others.

Instructions:

- **Be Creative and Diverse**: Think of various methods, theories, or perspectives that could lead to a solution to the <question>.
- **Clarity and Precision**: Ensure each task is clearly defined and focuses on a specific aspect of the problem.
- **Avoid Redundancy**: The tasks should cover different angles to avoid overlapping efforts.

Important: Do not generate more than {max_new_tasks} tasks.
"""


USER_PROMPT_INITIAL_SUB_TASK = """
You are presented with the following task delimited by the <Task> XML tags:
<Task>
{task}
</Task>
The task is aimed towards solving a part of the main question delimited by the <Question> XML tags.
<Question>
{question}
</Question>

Your job is to create one or more initial subtasks with short titles (maximum 4 words) that will serve as the root nodes of the task graph. \
These subtasks should represent different approaches to solve the task. Each task should be independent of other tasks.

Instructions:

- **Be Creative and Diverse**: Think of various methods, theories, or perspectives that could lead to a solution to the <Task>.
- **Clarity and Precision**: Ensure each subtask is clearly defined and focuses on a specific aspect of the problem.
- **Avoid Redundancy**: The tasks should cover different angles to avoid overlapping efforts.

Important: Do not generate more than {max_new_tasks} subtasks.
"""

USER_PROMPT_NEW_TASK = """
You are part of a reasoning system designed to solve complex problems by generating a dynamic graph of tasks.
You have already contributed by proposing initial tasks or strategies in earlier layers of the graph. 
Your goal is to assess and improve upon the existing task graph, and if needed, further refine the solution to the following question, delimited by the <Question> XML tags:
Ensure that you do not generate repetitive tasks and that each new task contributes meaningfully to the overall solution.
<Question>  
{question}  
</Question>

You are provided with the current state of the task graph, which includes answers and tasks proposed in previous layers. Make sure not to propose tasks that duplicate existing efforts.:

<TaskGraph>  
{parent_answers}  
</TaskGraph>

Additionally, you are provided with the strategies that guided the creation of each node in the task graph to help you understand the reasoning behind previous tasks:

<LayerStrategy>  
{layer_strategy}  
</LayerStrategy>

Based on the information above, you should:

1. **Review** the answers and strategies already proposed in the task graph.  
2. **Evaluate** whether the current task graph has already reached a conclusion or if further exploration or additional information is needed to resolve the question.  
   
   - If the current graph reaches a conclusion, you must propose a **Final Task** that consolidates the results and provides a direct answer to the question.  
     - Ensure that the **Final Task** references only unique parent node IDs, avoiding unnecessary repetitions.

   - If the graph is **incomplete**, you need to:  
     1. Identify why the graph is incomplete or insufficient to answer the question.  
     2. Propose **new tasks** that can help fill the gaps in the graph.  
     3. For each new task, provide a **detailed strategy** explaining why it is needed and how it will help solve the question.

3. **Task Constraints**:
   - Do not generate more than **{max_new_tasks}** tasks in this step.
   - Each new task should reference only **unique parent node IDs** from the NodeList (listed below). Do **not repeat any parent node ID** within a single task.
   - Ensure each task references **no more than {max_new_tasks} unique parent node IDs**.

### Node List

The following is a list of relevant nodes available to you. You can only choose from these when creating new tasks:

**Node List:**  
<NodeList>  
{node_list}  
</NodeList>

### Important Guidelines:

1. **No Repetitions**: Do not repeat any parent node ID within a single task.
2. **Stay Focused**: Keep the task generation focused on solving the original question. Avoid tangents or unnecessary tasks that do not contribute directly to the final answer.
3. **Clear Justifications**: If the graph is complete, explain why the question can be answered based on the current tasks, and justify that a **Final Task** can be proposed. Propoaw a **Final Task** that integrates the insights from the graph.
"""


FINAL_TASK_SYS_PROMPT = """
You are synthesizing the reasoning from the task graph to provide a comprehensive answer to the question.
Your goal is to produce a final task that integrates insights from previous tasks and leads directly to the solution.

Key objectives:

- **Integration**: Combine relevant findings from the task graph.
- **Clarity and Completeness**: Present the solution clearly and ensure all aspects of the question are addressed.
- **Justification**: Provide reasoning that supports the final answer.

The final task should conclude the reasoning process and provide the answer to the <Question>.
"""

FINAL_TASK_USER_PROMPT = """
You are concluding the reasoning process for the question delimited by the <Question> XML tags.
Your node list (only nodes at the current depth) is delimited by the <NodeList> XML tags.
Your task:
- Generate a final task that references all relevant parent task IDs from the <NodeList> above.
- Do not reference tasks from other depths.
- Integrate insights from those tasks.
- Include a detailed explanation leading to the final answer.

Instructions:

- **Comprehensive Solution**: Ensure the final task addresses the question fully.
- **Logical Reasoning**: Present a clear line of reasoning that justifies the answer.
- **Conclusion**: This task should conclude the task graph to answer the <Question>.

Provide the final answer within this task.

<Question>
{question}
</Question>

<NodeList>
{node_list}
</NodeList>
"""




# TODO: if task is complex, give the model more time to think with prompting. maybe the model can break down the tasks 

COMPLEXITY_CHECK_SYS_PROMPT = """
Determine if a task is complex based on the context provided.
If the graph of the current solution to the question is large, it is possible that the task is simple because \
the previous tasks likely solved most parts of the problem.
"""
# <task id={id} title={title}> task content here </task>
COMPLEXITY_CHECK_USER_PROMPT = """ 
Assess if the task demlimited by <task> XML tags is diffucult to solve.

If the task is difficult to solve and requires multiple steps to be executed set the is_complex parameter to True.
If the graph delmited by <graph> XML tags provides enough information to answer the question delimited by \
<question> XML tags, set the is_complex parameter to False. 

<question>
{question}
</question>

<task> 
ID: {task_id} 
Title: {task_title}
Content: {task_content}
</task>

<graph>
{main_graph}
</graph>
"""