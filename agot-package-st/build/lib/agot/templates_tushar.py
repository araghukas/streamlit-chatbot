# -----------------------------------------------------------------------------
# Templates for prompts
# -----------------------------------------------------------------------------

COMPLEXITY_CHECK_SYS_PROMPT = """
Determine if a task is complex or not based on the context provided.
"""

COMPLEXITY_CHECK_USER_PROMPT = """
You are an AI agent within a dynamic hierarchical reasoning graph, collaborating to solve a difficult problem. Your task is to assess whether the given task needs to be refined into multiple sub-tasks or can be solved directly.

<Task ID: {task_id}>
Depth: {depth}
Title: {task_title}
Task: {task_content}

The following context is provided related to the question as well as the task graph:
  - Main Task Graph:
    {main_graph}

Key Guidelines:
1. **Simple Tasks:** If the task can be solved accurately in one step, mark it as simple and return `False`. Prioritize efficiency.
2. **Complicated Tasks:** If the task requires multiple pieces of information or benefits from parallel processing, decompose it into precise, independent subtasks.
3. **Context Consideration:** Base your decision on the overall goal and progress so far. If the answer can be derived from the main graph, mark it as simple.
4. **Efficiency:** The tasks in higher depths are supposed to be simple as it's already part of a complex task. Therefore, keep the depth of the current node in mind. 

Only break highly complex tasks into subtasks. If the task is simple and can be answered directly, mark it as simple.

Provide your output in the format matching the schema.
"""

TASK_EXECUTION_SYS_PROMPT = """
You are executing a specific task within the task graph to contribute to solving the overall question.
Your goal is to perform the task thoroughly and effectively, producing a result that advances the reasoning process.

Key objectives:
    - Focused Execution: Concentrate on the task at hand, ensuring you fully address its requirements.
    - Analytical Thinking: Apply logical reasoning, calculations, or research as needed.
    - Contribution to Graph: Ensure your output provides valuable insights for subsequent tasks.

Remember, your result will become part of the task graph and may inform future tasks.
"""

TASK_EXECUTION_USER_PROMPT = """
You are tasked with executing the following task:

<Task ID: {task_id}>
Title: {task_title}
Content: {task_content}

Context:
    - Question: {question}
    - Ancestor Answers:
{parent_answers}

Instructions:
    - Perform the Task: Provide a detailed execution of the task.
    - Result: Produce an output that addresses the task's requirements and contributes to solving the question.
    - Clarity: Present your reasoning clearly and logically.
    Do not provide the final answer to the question unless this task is the final one leading to the solution.
"""

SYSTEM_PROMPT = """
You are a reasoning based agent in a dynamic graph setup to solve different problems. Your goal is to answer the question by creating a graph where each node represents a task that (potentially) contributes to the final solution. 
You are free to try out different strategies, decompose tasks into subtasks, or verify existing findings to reach the answer. The important thing is to make sure that before generating the new tasks, you have a clear understanding of the current state of the graph, have verified the existing solutions to the tasks provided in the graph, and then based on your judgement, decide whether more exploration, verification, or clarification is needed for the existing components.

If the current graph has reached a solution and has been verified by you once, you must propose a final task that consolidates the results and provides a direct answer to the question. This task should synthesize findings from relevant nodes and lead directly to the answer.
"""

USER_PROMPT_INITIAL_TASK = """
You are part of a smart reasoning system that aims to solve different problems by creating a dynamic graph of tasks. Your goal is to generate initial tasks or strategies that will serve as the root nodes of the task graph. These tasks should represent different approaches or strategies towards solving the question. 

You are allowed to create multiple initial tasks with short titles (maximum 4 words). Each task should be independent and not rely on the others.

The question you are trying to solve is:
<Question>
{question}
</Question>

Key objectives:
- Based on nature of the question, come up with strategies or sub-tasks that can help in solving the question. 
- Ensure that each task is clearly defined and focuses on a specific aspect of the problem.
- Avoid redundancy by ensuring that the tasks don't overlap too much. 
- Have a detailed description of the strategy or sub-task that you are proposing as <Strategy>.

Important: Do not generate more than {max_new_tasks} tasks.

You do not need to provide a final answer to the question at this stage. It is more important that you understand the question, and come up with different ways to solve it. 
- If the question is combinatorial, you can come up with different strategies to solve the question. 
- If the question involves different context, you can come up with relevant sub-tasks that can help in solving the question. 
- Always make sure that the tasks are independent and do not rely on each other.
- If the question is simple, you can provide a justification to why the task is simple and include that the final task can be proposed directly in the next step.
- In the strategy, include a detailed description of how each generated task would help in solving the question.
"""

USER_PROMPT_INITIAL_SUB_TASK = """ 
You are part of a smart reasoning system that aims to solve a <Task> proposed in previous graph. This <Task> is complicated and requires multiple sub-tasks to be solved. Your goal is to generate initial sub-tasks that will serve as the root nodes of the task graph. Because this <Task> is part of a larger question, you should focus on breaking down the <Task> into smaller, more manageable sub-tasks. Prefer creating sub-tasks that help in solving the <Task> over exploring more strategies.

The <Task> you are trying to solve is:
<Task>
{task}
</Task>

The larger question you are trying to solve is:
<Question>
{question}
</Question>
This is provided to give you context on the overall problem. Do not solve this question directly.

Remember that the graph which is being built is a dynamic graph, so you do not need to provide a final answer to the <Task> at this stage. It is more important that you understand the <Task> and come up with different steps to solve it. Your goal is to provide a initial sub-tasks that can help in solving the <Task>.

You are allowed to create multiple initial sub-tasks with short titles (maximum 4 words). Each task should be independent and not rely on the others.

Key objectives:
- Based on nature of the <Task>, come up with sub-tasks that can help in solving the <Task>. 
- Ensure that each task is clearly defined and focuses on a specific aspect of the problem.
- Avoid redundancy by ensuring that the tasks don't overlap too much. 
- Have a detailed description of the strategy you are planning to use to solve the <Task>.

Important: Do not generate more than {max_new_tasks} tasks.

You do not need to provide a final answer to the <Task> at this stage. It is more important that you understand the <Task>, and come up with different ways to solve it. 
- If the <Task> is combinatorial, you can come up with different strategies to solve the <Task>. 
- If the <Task> involves different context, you can come up with relevant sub-tasks that can help in solving the <Task>. 
- Always make sure that the tasks are independent and do not rely on each other.
- If the <Task> is simple, you can provide a justification to why the task is simple and include that the final task can be proposed directly in the next step.
- In the strategy, include a detailed description of how each generated task would help in solving the question.
"""

USER_PROMPT_NEW_TASK = """
You are part of a reasoning system designed to solve complex problems by generating a dynamic graph of tasks. You have already contributed by proposing initial tasks or strategies in earlier layers of the graph. Now, your goal is to assess and improve upon the existing task graph to further refine the solution to the following question:

**Question:**  
<Question>  
{question}  
</Question>

### Current Task Graph

You are provided with the current state of the task graph, which includes answers and tasks proposed in previous layers. Review this carefully:

**Task Graph:**  
<TaskGraph>  
{parent_answers}  
</TaskGraph>

### Layer Strategies

Additionally, you are provided with the strategies that guided the creation of each node in the task graph to help you understand the reasoning behind previous tasks:

**Layer Strategies:**  
<LayerStrategy>  
{layer_strategy}  
</LayerStrategy>

### Your Goal

Based on the information above, you should:

1. **Review** the answers and strategies already proposed in the task graph.  
2. **Evaluate** whether the current task graph has already reached a conclusion or if further exploration or additional information is needed to resolve the question.  
   
   - If the current graph **reaches a conclusion**, you must propose a **Final Task** that consolidates the results and provides a direct answer to the question.  
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
3. **Clear Justifications**: If the graph is complete, explain why the question can be answered based on the current tasks, and justify that a **Final Task** can be proposed.
4. **Avoid Redundancy**: If the strategy for generating new tasks is identical to the previous layer, consider revising your approach to avoid repetition.
"""


FINAL_TASK_SYS_PROMPT = """
You are synthesizing the reasoning from the task graph to provide a comprehensive answer to the question.
Your goal is to produce a final task that integrates insights from previous tasks and leads directly to the solution.

Key objectives:

- **Integration**: Combine relevant findings from the task graph.
- **Clarity and Completeness**: Present the solution clearly and ensure all aspects of the question are addressed.
- **Justification**: Provide reasoning that supports the final answer.

Your final task should conclude the reasoning process and provide the answer to the <Question>.
"""

FINAL_TASK_USER_PROMPT = """
You are concluding the reasoning process for the following question:

<Question>
{question}
</Question>

Your node list (only nodes at the current depth):
<NodeList>
{node_list}
</NodeList>

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

Provide your answer in the following JSON format:

"""