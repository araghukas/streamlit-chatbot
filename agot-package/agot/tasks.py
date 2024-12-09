from typing import List, Optional, Union, Any

from pydantic import BaseModel, Field


class Task(BaseModel):
    """Base model for tasks in the thought graph."""

    content: str = Field(..., description="Detailed description of the task.")
    title: str = Field(..., description="Short title of the task (max 4 words).")


class InitialTask(BaseModel):
    """Model for the initial tasks forming the root of the thought graph."""

    tasks: List[Task] = Field(
        ...,
        description="List of initial tasks.",
    )
    strategy: str = Field(
        default=..., description="Detailed strategy (one or more) to apprach the problem at hand. It should also include the reason for this strategy."
    )

# Use a meta agent to create the schema


class CheckComplex(BaseModel):
    """Model for checking task complexity."""

    is_complex: bool = Field(
        default=..., description="Indicates if the task is complex and requires further breakdown."
    )
    justification: str = Field(
        default=..., description="Reasoning for the complexity decision."
    )


class NewTask(Task):
    """Model for tasks leading to new tasks in the thought graph."""

    parent_id: List[int] = Field(
        ...,
        description="Unique parent ID(s) from the <NodeList>.",
    )


class FinalTask(NewTask):
    """Model for the final task in the thought <Graph>."""

    answer: str = Field(
        default=..., description="The final answer to the question."
    )

class MultiTaskResponse(BaseModel):
    """New tasks or the final task in the thought graph."""

    tasks: Union[List[NewTask], FinalTask] = Field(
        default=...,
        description="A list of new tasks or the final task to conclude the graph.",
    )
    strategy: str = Field(
        default=..., description="Detailed strategy used to generate the new tasks. Also include the reason for this strategy."
    )

class EvaluateTask(BaseModel):
    """Model for evaluating a task in the thought graph."""

    content: str = Field(
        default=..., description="The result after evaluating the task."
    )

class NodeData(BaseModel):
    id: int
    title: str
    content: str
    answer: Optional[Any]
    depth: int
    layer: int
    subgraph: str
    strategy: str
    children: List[int] = []

    class Config:
        arbitrary_types_allowed = True
    
    def model_dump_with_depth(self, max_depth=10, current_depth=0):
        # Base case to stop recursion if max depth is reached
        if current_depth >= max_depth:
            return {"id": self.id}  # Only serialize basic information
        
        obj_dict = self.model_dump()

        if "children" in obj_dict:
            # Serialize children recursively with depth check
            obj_dict["children"] = [
                child.model_dump_with_depth(max_depth=max_depth, current_depth=current_depth + 1) 
                for child in obj_dict["children"]
            ]

        return obj_dict


NodeData.update_forward_refs()

class FinalAnswer(BaseModel):
    """Model for the final answer, including the graph."""

    final_answer: Any 
    graph: List[NodeData]

    def dump_json(self):
        """ Convert the FinalAnswer object to JSON."""
        return self.model_dump_json()