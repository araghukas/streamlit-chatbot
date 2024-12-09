"""Main implementation of Adaptive Graph of Thought (AGoT) framework."""

import asyncio
import itertools
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
from multi_agent_llm.llm import LLMBase, OpenAILLM
from networkx.readwrite import json_graph
from pydantic import BaseModel
from rich.console import Console
from rich.text import Text
from tqdm import tqdm

from .tasks import (CheckComplex, EvaluateTask, FinalAnswer, FinalTask,
                    InitialTask, MultiTaskResponse, NewTask, NodeData, Task)
from .templates import (COMPLEXITY_CHECK_SYS_PROMPT,
                        COMPLEXITY_CHECK_USER_PROMPT, FINAL_TASK_SYS_PROMPT,
                        FINAL_TASK_USER_PROMPT, SYSTEM_PROMPT,
                        TASK_EXECUTION_SYS_PROMPT, TASK_EXECUTION_USER_PROMPT,
                        USER_PROMPT_INITIAL_SUB_TASK, USER_PROMPT_INITIAL_TASK,
                        USER_PROMPT_NEW_TASK)


class PromptCategory(Enum):
    """Categories of prompts used in the AGoT framework."""

    INITIAL_TASK = auto()
    INITIAL_SUB_TASK = auto()
    NEW_TASK = auto()
    TASK_EXECUTION = auto()
    COMPLEXITY_CHECK = auto()
    FINAL_TASK = auto()


@dataclass
class NodePosition:
    depth: int  # D#
    layer: int  # L#
    position: int  # P#

    def __str__(self) -> str:
        return f"D{self.depth}L{self.layer}P{self.position}"


class AGoTLogger:
    def __init__(
        self,
        verbose: int = 0,
        callback: Optional[Callable[[dict], None]] = None
    ):
        self.console = Console()
        self.verbose = verbose
        self.start_times = {}  # Store start times for each node
        # Use a self.print and have a switch to turn it off or on.

        self.callback = callback

    def start_timing(self, node_id: int):
        """Start timing for a specific node"""
        self.start_times[node_id] = perf_counter()

    def get_elapsed_time(self, node_id: int) -> float:
        """Get elapsed time for a node in seconds"""
        if node_id in self.start_times:
            return perf_counter() - self.start_times[node_id]
        return 0

    def format_time(self, seconds: float) -> str:
        """Format time duration with appropriate units"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"

    def get_node_lineage(self, node_id: int, dag: nx.DiGraph) -> str:
        """Construct the lineage string for a node based on depth"""
        lineage = []
        current_node_id = node_id
        # Change to for loop
        while True:
            node_data = dag.nodes[current_node_id]
            depth = node_data.get("depth", 0)
            layer = node_data.get("layer", 0)

            # Find position in layer
            same_layer_nodes = [
                n
                for n, d in dag.nodes(data=True)
                if d.get("depth") == depth and d.get("layer") == layer
            ]
            position = same_layer_nodes.index(current_node_id)

            current_pos = f"D{depth}L{layer}P{position}"
            lineage.insert(0, current_pos)

            predecessors = list(dag.predecessors(current_node_id))
            if not predecessors:
                break
            # Assuming single parent for simplicity
            current_node_id = predecessors[0]

        return "->".join(lineage)

    def log(
        self,
        level: int,
        message: str,
        title: Optional[str] = None,
        node_id: Optional[Union[int, List[int]]] = None,
        dag: Optional[nx.DiGraph] = None,
        content: Optional[str] = None,
        response: Optional[str] = None,
        context: Optional[dict] = None,
        is_complex: Optional[bool] = None,
    ):
        """Main logging method with rich formatting based on verbosity"""
        # Use a file to log the outputs and the time it took to run the code.
        if self.verbose < level:
            return

        if node_id is not None and dag is not None:
            if isinstance(node_id, list):
                self.console.print(
                    f"Node_Ids: {node_id} \nContext:\n{json.dumps(context, indent=2)}\n", style="italic"
                )
                return
            lineage = self.get_node_lineage(node_id, dag)
        elif dag is not None:
            lineage = "NEW_TASK"
        else:
            lineage = "ROOT"

        if level == 1:
            styled_text = Text()
            styled_text.append(f"[{lineage}] ", style="cyan")
            styled_text.append(f"Node {node_id}: ", style="yellow")
            styled_text.append(title or message, style="white")

            if is_complex is not None:
                complexity_style = "red" if is_complex else "green"
                complexity_text = " (complex)" if is_complex else " (simple)"
                styled_text.append(complexity_text, style=complexity_style)

            if node_id in self.start_times:
                elapsed_time = self.get_elapsed_time(node_id)
                time_str = self.format_time(elapsed_time)
                styled_text.append(f" [{time_str}]", style="bright_black")

            self.console.print(styled_text)

        elif level == 2:
            styled_text = Text()
            styled_text.append(f"[{lineage}] ", style="cyan")
            styled_text.append(f"Node {node_id}: ", style="yellow")
            styled_text.append(title or message, style="white")

            if is_complex is not None:
                complexity_style = "red" if is_complex else "green"
                complexity_text = " (complex)" if is_complex else " (simple)"
                styled_text.append(complexity_text, style=complexity_style)

            if node_id in self.start_times:
                elapsed_time = self.get_elapsed_time(node_id)
                time_str = self.format_time(elapsed_time)
                styled_text.append(f" [{time_str}]", style="bright_black")

            self.console.print(styled_text)

            if content:
                self.console.print(f"Task Content:\n{content}\n", style="dim")
            if response:
                self.console.print(
                    f"LLM Response:\n{response}\n", style="bold")

        elif level >= 3:
            self.log(
                2, message, title, node_id, dag, content, response, context, is_complex
            )
            if context:
                self.console.print(
                    f"Context:\n{json.dumps(context, indent=2)}\n", style="italic"
                )

    def callback(self, *args, node_id, **kwargs) -> Optional[Any]:
        if self.callback is not None:
            return self.callback(*args, **kwargs)
        return None


class AGoT:
    """Adaptive Graph of Thought Framework.

    An LLM reasoning framework that starts with initial tasks, execute them, dynamically generates new tasks in two dimensions, and evaluates them to get a final answer.
    """

    _prompt_templates = {
        PromptCategory.INITIAL_TASK: (
            SYSTEM_PROMPT,
            USER_PROMPT_INITIAL_TASK,
        ),
        PromptCategory.NEW_TASK: (
            SYSTEM_PROMPT,
            USER_PROMPT_NEW_TASK,
        ),
        PromptCategory.INITIAL_SUB_TASK: (
            SYSTEM_PROMPT,
            USER_PROMPT_INITIAL_SUB_TASK,
        ),
        PromptCategory.TASK_EXECUTION: (
            TASK_EXECUTION_SYS_PROMPT,
            TASK_EXECUTION_USER_PROMPT,
        ),
        PromptCategory.COMPLEXITY_CHECK: (
            COMPLEXITY_CHECK_SYS_PROMPT,
            COMPLEXITY_CHECK_USER_PROMPT,
        ),
        PromptCategory.FINAL_TASK: (
            FINAL_TASK_SYS_PROMPT,
            FINAL_TASK_USER_PROMPT,
        ),
    }

    def __init__(
        self,
        llm: LLMBase,
        max_new_tasks: int = 3,
        max_depth: int = 1,
        verbose: int = 0,
        max_num_layers: int = 2,
        max_concurrent_tasks: int = 500,
        layer_depth_reduction: bool = False,
        log_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.llm = llm
        self.max_new_tasks = max_new_tasks
        self.verbose = verbose
        self.max_depth = max_depth
        self.max_num_layers = max_num_layers
        self.max_concurrent_tasks = max_concurrent_tasks
        self.layer_depth_reduction = layer_depth_reduction

        self._id_counter = itertools.count(0)
        self._event_log = []
        self.user_schema = None
        self.logger = AGoTLogger(verbose, callback=log_callback)
        # Initialize the semaphore for limiting concurrent tasks
        self.semaphore = asyncio.Semaphore(
            self.max_concurrent_tasks)  # New line

    def get_task_string(self, dag: nx.DiGraph, node_id: int) -> str:
        """Get the string representation of a task by its node ID."""
        return dag.nodes[node_id]["task"]

    def get_node_list(self, dag: nx.DiGraph, depth: int = None) -> str:
        """Get a formatted node list with node IDs and titles in the dag at a given depth."""
        nodes = []
        for node_id in dag.nodes:
            node_data = dag.nodes[node_id]
            if depth is None or node_data.get("depth") == depth:
                title = node_data.get("title", "")
                nodes.append(
                    f"<ID> {node_id} </ID>/n <Title>: {title}</Title>")
        return "\n".join(nodes)

    def _add_task(
        self,
        task: Task,
        parent_id: Union[int, List[int]] = None,
        dag: nx.DiGraph = None,
        depth: int = 0,
        layer: int = 0,
        strategy: str = None,
    ) -> Tuple[int, nx.DiGraph]:
        """Add a task to the graph as a node."""
        _id = next(self._id_counter)
        task_string = self._format_task(_id, task)
        if dag is not None:
            dag.add_node(
                _id, task=task_string, depth=depth, layer=layer, title=task.title, strategy=strategy
            )
            if parent_id is not None:
                if not isinstance(parent_id, list):
                    parent_id = [parent_id]
                parent_id = list(set(parent_id).intersection(
                    set(list(dag.nodes))))  # Ensure unique parent IDs
                if not parent_id:
                    warnings.warn(
                        "No common parents found in the parent graph when adding a task."
                    )
                for _parent_id in parent_id:
                    if _id != _parent_id and not dag.has_edge(_parent_id, _id):
                        dag.add_edge(_parent_id, _id)

        self.logger.callback("add_task", node_id=_id, dag=dag, depth=depth, layer=layer)
        return _id, dag

    def _format_task(self, _id: int, task: Task) -> str:
        """Format a task as a string with a unique ID."""
        return f"<{_id}>[{task.title}] {task.content}</{_id}>"

    def _format_strategy(self, strategy: Dict[int, str]) -> str:
        """Format the strategy dictionary as a string."""
        return json.dumps(strategy)

    def run(self, question: str, schema: BaseModel = None) -> Optional[FinalAnswer]:
        """Synchronously run the AGoT framework to generate a response to the given question."""
        return asyncio.run(self.run_async(question, schema))

    async def run_async(
        self, question: str, schema: BaseModel = None
    ) -> FinalAnswer:
        self.user_schema = schema
        dag = nx.DiGraph()
        final_response = None
        if self.verbose >= 4:
            self._log(4, "Starting task processing")

        initial_tasks, strategy = await self._generate_initial_tasks(question)
        depth = 0
        layer = 0
        layer_dict = {}
        layer_dict[layer] = []

        # Create a dictionary of strategies with layers as keys
        strategy_dict = {layer: strategy}
        for task in initial_tasks:
            node_id, _ = self._add_task(
                task, dag=dag, depth=depth, layer=layer, parent_id=None, strategy=strategy
            )
            layer_dict[layer].append((node_id, task))

        for layer in range(self.max_num_layers):
            if self.verbose >= 3:
                self._log(
                    3,
                    "Processing tasks in graph",
                    node_id=(list(dag.nodes)[0] if dag.nodes else None),
                    dag=dag,
                )

            tasks = [
                self.process_task(
                    node_id=node_id,
                    task=task,
                    question=question,
                    depth=depth,
                    layer=layer,
                    dag=dag,
                    main_dag=dag,
                )
                for node_id, task in layer_dict[layer]
            ]
            # Limit the number of concurrent tasks
            async with self.semaphore:
                await asyncio.gather(*tasks)

            node_list_str = self.get_node_list(dag=dag, depth=depth)
            new_tasks, strategy = await self._generate_new_tasks(
                question=question,
                depth=depth,
                node_list_str=node_list_str,
                parent_answers=self.graph_to_string(dag=dag),
                dag=dag,
                strategy_dict=strategy_dict,
            )
            strategy_dict[layer+1] = strategy
            if isinstance(new_tasks[0], FinalTask):
                new_tasks = new_tasks[0]
                self._log(1, f"Final answer reached at Layer {layer}")
                final_node_id, _ = self._add_task(
                    new_tasks,
                    dag=dag,
                    depth=depth,
                    layer=layer,
                    parent_id=new_tasks.parent_id,
                    strategy=strategy,
                )
                final_response = await self._evaluate_task(
                    task_id=final_node_id,
                    task=new_tasks,
                    question=question,
                    final_schema=self.user_schema,
                    dag=dag,
                )

                dag.nodes[final_node_id]["answer"] = final_response
                self.logger.callback("answer", node_id=final_node_id, dag=dag, depth=depth, final=True)

                return FinalAnswer(
                    final_answer=final_response,
                    graph=self.export_graph(dag),
                )

            if layer < self.max_num_layers-1:
                layer_dict[layer+1] = []
                for new_task in new_tasks[:self.max_new_tasks]:
                    node_id, _ = self._add_task(
                        new_task,
                        dag=dag,
                        depth=depth,
                        layer=layer+1,
                        parent_id=new_task.parent_id,
                        strategy=strategy,
                    )
                    layer_dict[layer+1].append((node_id, new_task))

        if final_response is None:
            node_list_str = self.get_node_list(dag=dag, depth=depth)
            final_task = await self._generate_final_task(
                question=question,
                depth=depth,
                node_list_str=node_list_str,
                parent_answers=self.format_all_answers(depth=depth, dag=dag),
                dag=dag,
            )

            parent_ids = [
                node_id
                for node_id in dag.nodes
                if dag.nodes[node_id].get("depth") == 0
            ]
            final_node_id, _ = self._add_task(
                final_task,
                dag=dag,
                depth=depth,
                layer=self.max_num_layers + 1,
                parent_id=parent_ids,
                strategy=strategy,
            )
            final_answer = await self._evaluate_task(
                task_id=final_node_id,
                task=final_task,
                question=question,
                final_schema=self.user_schema,
                dag=dag,
            )

            dag.nodes[final_node_id]["answer"] = final_answer
            self.logger.callback("answer", node_id=final_node_id, dag=dag, depth=depth, final=True)
            return FinalAnswer(
                final_answer=final_answer,
                graph=self.export_graph(dag),
            )

    def format_all_answers(self, depth: int, dag: nx.DiGraph) -> str:
        """Format all answers at a given depth into a string."""
        answers = []
        for node_id in dag.nodes:
            node = dag.nodes[node_id]
            if node.get("depth", 0) == depth and "answer" in node:
                answers.append(
                    f"<{node_id}>\nTitle: {node['title']}\nAnswer: {node['answer']}\n</{node_id}>"
                )
        if not answers:
            return "No answers available."
        return "\n".join(answers)

    async def _generate_initial_tasks(self, question: str) -> Tuple[List[Task], str]:
        response = await self._generate(
            category=PromptCategory.INITIAL_TASK,
            schema=InitialTask,
            question=question,
            max_new_tasks=self.max_new_tasks,
            dag=None,
        )
        return [response.tasks, response.strategy]

    async def _generate_initial_sub_tasks(
        self, task: Task, question: str, dag: nx.DiGraph = None
    ) -> Tuple[List[Task], str]:
        response = await self._generate(
            category=PromptCategory.INITIAL_SUB_TASK,
            schema=InitialTask,
            question=question,
            max_new_tasks=self.max_new_tasks,
            task=task,
            dag=dag,
        )
        return response.tasks, response.strategy

    async def process_task(
        self,
        node_id: int,
        task: Task,
        question: str,
        depth: int,
        layer: int,
        dag: nx.DiGraph,
        main_dag: nx.DiGraph,
    ) -> None:
        async with self.semaphore:
            self.logger.start_timing(node_id)

            complexity_check = await self._check_complex(
                question=question,
                task_id=node_id,
                task=task,
                dag=dag,
                main_graph=self.graph_to_string(main_dag),
                depth=depth,
            )
            is_complex = complexity_check.is_complex

            if self.verbose >= 1:
                self._log(1, task.title, node_id=node_id,
                          dag=dag, is_complex=is_complex)

            if is_complex and depth < self.max_depth:
                self.logger.callback("is_complex", node_id=node_id, dag=dag, depth=depth)
                subgraph_answer, subgraph = await self.process_subgraph(
                    task=task,
                    question=question,
                    node_id=node_id,
                    depth=depth + 1,
                    dag=dag,
                    main_dag=main_dag,
                )
                dag.nodes[node_id]["answer"] = subgraph_answer
                dag.nodes[node_id]["subgraph"] = subgraph
                self.logger.callback("update_complex", node_id=node_id, dag=dag, depth=depth, sub_dag=subgraph)
            else:
                task_answer = await self._evaluate_task(
                    task_id=node_id,
                    task=task,
                    question=question,
                    final_schema=None,
                    dag=dag,
                )

                if self.verbose >= 2:
                    self._log(
                        2,
                        task.title,
                        node_id=node_id,
                        dag=dag,
                        content=task.content,
                        response=str(task_answer),
                    )
                dag.nodes[node_id]["answer"] = task_answer
                self.logger.callback("answer", node_id=node_id, dag=dag, depth=depth)

            self.logger.start_times[node_id] = 0  # Reset timing after logging

    async def process_subgraph(
        self,
        task: Task,
        question: str,
        node_id: int,
        depth: int,
        dag: nx.DiGraph,
        main_dag: nx.DiGraph,
    ) -> Tuple[str, nx.DiGraph]:
        if self.verbose >= 4:
            self._log(4, "Processing subgraph", node_id=node_id, dag=main_dag)

        subgraph = nx.DiGraph()
        self.logger.callback("new_subgraph", node_id=node_id, dag=subgraph, depth=depth)
        final_response = None
        layer = 0
        layer_dict = {}
        layer_dict[layer] = []
        initial_tasks, strategy = await self._generate_initial_sub_tasks(task, question)
        for task in initial_tasks:
            subgraph_node_id, _ = self._add_task(
                task,
                dag=subgraph,
                depth=depth,
                layer=layer,
                strategy=strategy,
            )
            layer_dict[layer].append((subgraph_node_id, task))
        # Create a strategy dictionary for the subgraph
        strategy_dict = {layer: strategy}
        max_layers = self.max_num_layers - \
            depth if self.layer_depth_reduction else self.max_num_layers
        for layer in range(max_layers):
            if self.verbose >= 3:
                self._log(
                    3,
                    "Processing subgraph tasks",
                    node_id=node_id,
                    dag=main_dag,
                    context={"subgraph": self.graph_to_string(dag=subgraph)},
                )

            tasks = [
                self.process_task(
                    node_id=node_sub_id,
                    task=task,
                    question=question,
                    depth=depth,
                    layer=layer,
                    dag=subgraph,
                    main_dag=main_dag,
                )
                for node_sub_id, task in layer_dict[layer]
            ]
            async with self.semaphore:
                await asyncio.gather(*tasks)

            node_list_str = self.get_node_list(dag=subgraph, depth=depth)
            new_tasks, strategy = await self._generate_new_tasks(
                question=task.content,
                depth=depth,
                node_list_str=node_list_str,
                parent_answers=self.graph_to_string(dag=subgraph),
                dag=subgraph,
                strategy_dict=strategy_dict,
            )
            strategy_dict[layer+1] = strategy

            if isinstance(new_tasks[0], FinalTask):
                new_tasks = new_tasks[0]
                self._log(1, f"Final answer reached at Layer {layer}")
                final_node_id, _ = self._add_task(
                    new_tasks,
                    dag=subgraph,
                    depth=depth,
                    layer=layer,
                    parent_id=[
                        node_sub_id
                        for node_sub_id in subgraph.nodes
                        if subgraph.nodes[node_sub_id].get("depth", 0) == depth
                    ],
                    strategy=strategy,
                )
                final_response = await self._evaluate_task(
                    task_id=final_node_id,
                    task=new_tasks,
                    question=question,
                    dag=subgraph,
                )

                subgraph.nodes[final_node_id]["answer"] = final_response.content
                self.logger.callback("answer", node_id=final_node_id, dag=subgraph, depth=depth, final=True)
                return final_response, subgraph
            if layer < self.max_num_layers-depth-1:
                layer_dict[layer+1] = []
                for new_task in new_tasks[:self.max_new_tasks]:
                    node_sub_id, _ = self._add_task(
                        new_task,
                        dag=subgraph,
                        depth=depth,
                        layer=layer+1,
                        parent_id=new_task.parent_id,
                        strategy=strategy,
                    )
                    layer_dict[layer+1].append((node_sub_id, new_task))

        if final_response is None:
            node_list_str = self.get_node_list(dag=subgraph, depth=depth)
            final_task = await self._generate_final_task(
                question=question,
                depth=depth,
                node_list_str=node_list_str,
                parent_answers=self.format_all_answers(
                    depth=depth, dag=subgraph),
                dag=subgraph,
            )

            parent_ids = [
                node_sub_id
                for node_sub_id in subgraph.nodes
                if subgraph.nodes[node_sub_id].get("depth", 0) == depth
            ]
            final_node_id, _ = self._add_task(
                final_task,
                dag=subgraph,
                depth=depth,
                layer=self.max_num_layers + 1,
                parent_id=parent_ids,
                strategy=strategy,
            )
            final_answer = await self._evaluate_task(
                task_id=final_node_id,
                task=final_task,
                question=question,
                final_schema=None,
                dag=subgraph,
            )

            subgraph.nodes[final_node_id]["answer"] = final_answer.content
            self.logger.callback("answer", node_id=final_node_id, dag=subgraph, depth=depth, final=True)
            return final_answer, subgraph

    def _log(
        self,
        level: int,
        message: str,
        depth=None,
        layer=None,
        node_id=None,
        content=None,
        response=None,
        context=None,
        dag=None,
        is_complex=None,
    ):
        """Enhanced logging with rich formatting"""
        if self.verbose < level:
            return

        title = message
        if node_id is not None and dag is not None:
            self.logger.log(
                level=level,
                message=message,
                title=title,
                node_id=node_id,
                dag=dag,
                content=content,
                response=response,
                context=context,
                is_complex=is_complex,
            )
        elif dag is not None:
            self.logger.log(
                level=level,
                message=message,
                title=title,
                dag=dag,
                content=content,
                response=response,
                context=context,
                is_complex=is_complex,
            )
        else:
            self.logger.log(level=level, message=message)
        self._event_log.append(message)

    async def _generate_final_task(
        self,
        question: str,
        depth: int,
        node_list_str: str,
        parent_answers: str,
        dag: nx.DiGraph,
        task: Task = None,
    ) -> FinalTask:
        """Generate a final task when max depth or layers are reached."""
        response = await self._generate(
            category=PromptCategory.FINAL_TASK,
            schema=FinalTask,
            question=question,
            task=task,
            depth=depth,
            max_new_tasks=1,
            node_list=node_list_str,
            parent_answers=parent_answers,
            force_final_task=True,
            dag=dag,
        )
        if isinstance(response, FinalTask):
            return response
        if isinstance(response.tasks, FinalTask):
            return response.tasks
        elif isinstance(response.tasks, list) and isinstance(
            response.tasks[0], FinalTask
        ):
            return response.tasks[0]
        else:
            raise ValueError("Failed to generate a final task.")

    def get_ancestors_answers(self, node_id: int, dag: nx.DiGraph) -> Dict[int, Any]:
        """Get the answers of the ancestor nodes of the given node."""
        # parent_answers = {}
        # for parent_id in nx.ancestors(dag, node_id):
        #     parent_node = dag.nodes[parent_id]
        #     if "answer" in parent_node:
        #         parent_answers[parent_id] = parent_node["answer"]

        # return parent_answers

        return self.graph_to_string(dag.subgraph(list(nx.ancestors(dag, node_id))))

    def format_parent_answers(self, parent_answers: Dict[int, Any]) -> str:
        """Format parent answers into a string for inclusion in prompts."""
        if not parent_answers:
            return "No parent answers available."
        lines = []
        for parent_id, answer in parent_answers.items():
            lines.append(
                f"<Parent ID: {parent_id}>\nAnswer: {answer}\n</Parent ID: {parent_id}>"
            )
        return "\n".join(lines)

    async def _generate_new_tasks(
        self,
        question: str,
        depth: int,
        node_list_str: str,
        parent_answers: str,
        strategy_dict: Dict[int, str],
        force_final_task: bool = False,
        dag: nx.DiGraph = None,
    ) -> Tuple[List[Task], str]:
        response = await self._generate(
            category=PromptCategory.NEW_TASK,
            schema=MultiTaskResponse,
            question=question,
            depth=depth,
            max_new_tasks=self.max_new_tasks - depth,
            node_list=node_list_str,
            parent_answers=parent_answers,
            force_final_task=force_final_task,
            dag=dag,
            layer_strategy=self._format_strategy(strategy_dict),
        )
        if isinstance(response.tasks, FinalTask):
            response.tasks = [response.tasks]
        self._log(
            3,
            "Generated new tasks",
            depth=depth,
            node_id=(list(dag.nodes) if dag.nodes else None),
            dag=dag,
            context={"new_tasks": [str([list(set(task.parent_id))])
                                   for task in response.tasks]},
        )
        return response.tasks, response.strategy

    async def _evaluate_task(
        self,
        task_id: int,
        task: Task,
        question: str,
        final_schema: BaseModel = None,
        dag: nx.DiGraph = None,
    ) -> Any:

        lineage_tasks = self.get_node_lineage_tasks(task_id, dag)
        lineage_str = "->".join(
            [
                f"D{node_data.get('depth')}L{node_data.get('layer')}P{node_data.get('position')}"
                for node_data in lineage_tasks
            ]
        )

        parent_answers_str = self.get_ancestors_answers(task_id, dag)

        response = await self._generate(
            category=PromptCategory.TASK_EXECUTION,
            schema=final_schema if final_schema else EvaluateTask,
            dag=dag,
            task_id=task_id,
            task_title=task.title,
            task_content=task.content,
            question=question,
            lineage=lineage_str,
            parent_answers=parent_answers_str,
        )

        return response

    def get_node_lineage_tasks(
        self, node_id: int, dag: nx.DiGraph
    ) -> List[Dict[str, Any]]:
        """Get the lineage of tasks leading to the current node based on depth."""
        lineage = []
        current_node_id = node_id
        # Change to for loop
        while True:
            node_data = dag.nodes[current_node_id]
            lineage.append(node_data)
            predecessors = list(dag.predecessors(current_node_id))
            if not predecessors:
                break
            current_node_id = predecessors[0]
        lineage.reverse()
        return lineage

    async def _check_complex(
        self,
        question: str,
        task_id: int,
        task: Task,
        depth: int = 0,
        dag: nx.DiGraph = None,
        main_graph: str = "",
    ) -> CheckComplex:
        response = await self._generate(
            category=PromptCategory.COMPLEXITY_CHECK,
            schema=CheckComplex,
            task_id=task_id,
            task_title=task.title,
            task_content=task.content,
            main_graph=main_graph,
            dag=dag,
            depth=depth,
            question=question,
        )
        return response

    async def _generate(
        self,
        category: PromptCategory,
        schema: BaseModel,
        dag: nx.DiGraph = None,
        **user_kwargs,
    ) -> BaseModel:
        force_final_task = user_kwargs.pop("force_final_task", False)
        if force_final_task:
            category = PromptCategory.FINAL_TASK

        system_prompt, user_prompt_template = self._prompt_templates[category]
        system_prompt = system_prompt.format(**user_kwargs)
        user_prompt = user_prompt_template.format(**user_kwargs)
        messages = self.llm.format_prompt(system_prompt, user_prompt)

        if self.verbose >= 3:
            context = {
                "system_prompt": messages[0]["content"],
                "user_prompt": messages[1]["content"],
            }
            self._log(
                3,
                f"Generation for {category.name}",
                node_id=user_kwargs.get("task_id"),
                dag=dag,
                context=context,
            )

        response = await self.llm.generate_async(messages, schema=schema)

        if self.verbose >= 3:
            self._log(
                3,
                f"Response for {category.name}",
                node_id=user_kwargs.get("task_id"),
                dag=dag,
                response=str(response),
            )
        return response

    def graph_to_string(self, dag: nx.DiGraph) -> str:
        """Generate an XML representation of the graph."""

        def node_to_xml(node_id, visited, dag):
            if node_id in visited:
                return ""
            visited.add(node_id)
            node_data = dag.nodes[node_id]
            title = node_data.get("title", "")
            content = node_data.get("task", "")
            answer = node_data.get("answer", "")
            depth = node_data.get("depth", 0)
            layer = node_data.get("layer", 0)
            subgraph = node_data.get("subgraph", "")
            strategy = node_data.get("strategy", "")
            xml = f'<Node title="{title}" depth="{depth}" layer="{layer}">\n'
            xml += f"  <Content>{content}</Content>\n"
            xml += f"  <Strategy>{strategy}</Strategy>\n"
            if answer:
                xml += f"  <Answer>{answer}</Answer>\n"
            if subgraph:
                xml += "  <Subgraph>\n"
                for sub_node_id in subgraph:
                    sub_xml = subgraph.nodes[sub_node_id].get("title", "")
                    xml += sub_xml
                xml += "  </Subgraph>\n"
            children = list(dag.successors(node_id))
            if children:
                xml += "  <Children>\n"
                for child_id in children:
                    child_xml = dag.nodes[child_id].get("title", "")
                    xml += child_xml
                xml += "  </Children>\n"
            xml += "</Node>\n"
            return xml

        visited = set()
        xml = ""
        nodes = [n for n, _ in dag.in_degree()]
        for node in nodes:
            xml += node_to_xml(node, visited, dag)
        return xml

    def export_graph(self, dag: nx.DiGraph) -> List[NodeData]:
        """Export the graph as a list of NodeData objects."""
        node_dict = {}
        for node_id in dag.nodes:
            node_data = dag.nodes[node_id]

            # Add subgraph
            subgraph = dag.nodes[node_id].get("subgraph", None)
            # Convert to a JSON-compatible dictionary
            # data = json_graph.node_link_data(subgraph) if subgraph else None

            # Serialize to JSON string
            # json_string = json.dumps(data)
            json_string = self.graph_to_string(subgraph) if subgraph else ""

            node_dict[node_id] = NodeData(
                id=node_id,
                title=node_data.get("title", ""),
                content=node_data.get("task", ""),
                answer=node_data.get("answer", None),
                depth=node_data.get("depth", 0),
                layer=node_data.get("layer", 0),
                subgraph=json_string,
                strategy=node_data.get("strategy", ""),
                children=[],
            )

        # Set the children relationships
        for node_id in dag.nodes:
            node_dict[node_id].children = [
                child_id for child_id in dag.successors(node_id)
            ]

        return [node_dict[node_id] for node_id in dag.nodes]


def run_AGoT_batched(
    questions: List[str],
    model_name: str = "gpt-4o-mini",
    batch_size: int = 6,
    max_new_thoughts: int = 3,
    max_num_layers=2,
    verbose: bool = False,
    max_depth: int = 1,
) -> List[FinalTask]:
    """Run the AGoT framework on a batch of questions."""

    if not isinstance(questions, list):
        raise ValueError("Questions must be submitted as a list.")

    llm = OpenAILLM(model_name)

    async def _AGoT_run_async(q):
        agent = AGoT(
            llm=llm,
            max_new_thoughts=max_new_thoughts,
            max_num_layers=max_num_layers,
            verbose=verbose,
            max_depth=max_depth,
        )
        result = await agent.run_async(q)
        graph = agent.return_graph()
        return result, graph

    final_responses = []
    graphs = []

    for i in tqdm(range(0, len(questions), batch_size)):
        questions_batch = questions[i: i + batch_size]
        tasks = [_AGoT_run_async(q) for q in questions_batch]

        responses_batch = asyncio.run(asyncio.gather(*tasks))

        responses_batch, graphs_batch = zip(*responses_batch)

        final_responses.extend(responses_batch)
        graphs.extend(graphs_batch)

    return final_responses, graphs
