"""
Pipeline - Data flow between agents/skills
"""

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio


@dataclass
class PipelineStage:
    """
    PipelineStage - A single stage in a data pipeline.
    
    Each stage transforms input data and passes it to the next stage.
    
    Example:
        >>> stage = PipelineStage(
        ...     name="parse",
        ...     processor=lambda data: json.loads(data),
        ... )
    """
    
    name: str
    processor: Callable[[Any], Any]
    validator: Optional[Callable[[Any], bool]] = None
    error_handler: Optional[Callable[[Exception], Any]] = None
    
    async def process(self, data: Any) -> Any:
        """Process data through this stage"""
        # Validate input if validator exists
        if self.validator and not self.validator(data):
            raise ValueError(f"Validation failed for stage '{self.name}'")
        
        try:
            result = self.processor(data)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            if self.error_handler:
                return self.error_handler(e)
            raise


class Pipeline:
    """
    Pipeline - A data processing pipeline.
    
    Pipelines transform data through a series of stages, where each
    stage's output becomes the next stage's input. Unlike workflows,
    pipelines focus on data transformation rather than agent orchestration.
    
    Example:
        >>> pipeline = (Pipeline("TextProcessing")
        ...     .add_stage(PipelineStage("clean", clean_text))
        ...     .add_stage(PipelineStage("tokenize", tokenize))
        ...     .add_stage(PipelineStage("embed", embed))
        ... )
        >>> embeddings = await pipeline.run("raw text input")
    """
    
    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.stages: List[PipelineStage] = []
        self._middleware: List[Callable] = []
    
    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """Add a stage to the pipeline"""
        self.stages.append(stage)
        return self
    
    def add_transform(self, name: str, processor: Callable) -> "Pipeline":
        """Add a simple transform stage"""
        return self.add_stage(PipelineStage(name=name, processor=processor))
    
    def use(self, middleware: Callable) -> "Pipeline":
        """Add middleware that runs around each stage"""
        self._middleware.append(middleware)
        return self
    
    async def run(self, data: Any) -> Any:
        """
        Run data through the pipeline.
        
        Args:
            data: Input data to process
            
        Returns:
            Transformed output data
        """
        current_data = data
        
        for stage in self.stages:
            # Apply middleware
            for middleware in self._middleware:
                current_data = await self._apply_middleware(
                    middleware, stage, current_data
                )
            
            # Process stage
            current_data = await stage.process(current_data)
        
        return current_data
    
    async def _apply_middleware(
        self,
        middleware: Callable,
        stage: PipelineStage,
        data: Any
    ) -> Any:
        """Apply middleware to a stage execution"""
        result = middleware(stage, data)
        if asyncio.iscoroutine(result):
            result = await result
        return result if result is not None else data
    
    def branch(
        self,
        condition: Callable[[Any], bool],
        if_true: "Pipeline",
        if_false: Optional["Pipeline"] = None,
    ) -> "Pipeline":
        """Add a conditional branch to the pipeline"""
        async def branch_processor(data: Any) -> Any:
            if condition(data):
                return await if_true.run(data)
            elif if_false:
                return await if_false.run(data)
            return data
        
        return self.add_stage(PipelineStage(
            name="branch",
            processor=branch_processor,
        ))
    
    def parallel(self, *pipelines: "Pipeline") -> "Pipeline":
        """Run multiple pipelines in parallel on the same input"""
        async def parallel_processor(data: Any) -> List[Any]:
            tasks = [pipeline.run(data) for pipeline in pipelines]
            return await asyncio.gather(*tasks)
        
        return self.add_stage(PipelineStage(
            name="parallel",
            processor=parallel_processor,
        ))
    
    def __repr__(self) -> str:
        stage_names = [s.name for s in self.stages]
        return f"Pipeline({self.name}, stages={stage_names})"


class AgentPipeline(Pipeline):
    """
    AgentPipeline - A pipeline specifically for chaining agents.
    
    Each stage is an agent that processes and passes results to the next.
    
    Example:
        >>> pipeline = (AgentPipeline("ContentCreation")
        ...     .add_agent("research", researcher_agent)
        ...     .add_agent("write", writer_agent)
        ...     .add_agent("edit", editor_agent)
        ... )
        >>> final_content = await pipeline.run("Write about AI")
    """
    
    def add_agent(
        self,
        name: str,
        agent: Any,
        input_key: str = "message",
        output_key: str = "content",
    ) -> "AgentPipeline":
        """Add an agent as a pipeline stage"""
        async def agent_processor(data: Any) -> Any:
            # Handle different input formats
            if isinstance(data, dict):
                message = data.get(input_key, str(data))
            else:
                message = str(data)
            
            result = await agent.run(message)
            
            # Handle different output formats
            if hasattr(result, output_key):
                return getattr(result, output_key)
            elif hasattr(result, 'content'):
                return result.content
            return result
        
        return self.add_stage(PipelineStage(
            name=name,
            processor=agent_processor,
        ))
    
    def add_skill(
        self,
        name: str,
        skill: Any,
        **skill_params
    ) -> "AgentPipeline":
        """Add a skill as a pipeline stage"""
        async def skill_processor(data: Any) -> Any:
            # Merge input data with fixed params
            params = {**skill_params}
            if isinstance(data, dict):
                params.update(data)
            else:
                params["input"] = data
            
            result = await skill.execute(**params)
            return result.data if hasattr(result, 'data') else result
        
        return self.add_stage(PipelineStage(
            name=name,
            processor=skill_processor,
        ))


class DataPipeline(Pipeline):
    """
    DataPipeline - A pipeline for data transformation.
    
    Provides common data transformation utilities.
    """
    
    def map(self, func: Callable) -> "DataPipeline":
        """Map a function over a list of items"""
        async def map_processor(data: List[Any]) -> List[Any]:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.gather(*[func(item) for item in data])
            return [func(item) for item in data]
        
        return self.add_stage(PipelineStage(name="map", processor=map_processor))
    
    def filter(self, predicate: Callable[[Any], bool]) -> "DataPipeline":
        """Filter items based on a predicate"""
        def filter_processor(data: List[Any]) -> List[Any]:
            return [item for item in data if predicate(item)]
        
        return self.add_stage(PipelineStage(name="filter", processor=filter_processor))
    
    def reduce(self, func: Callable[[Any, Any], Any], initial: Any = None) -> "DataPipeline":
        """Reduce a list to a single value"""
        def reduce_processor(data: List[Any]) -> Any:
            from functools import reduce
            if initial is not None:
                return reduce(func, data, initial)
            return reduce(func, data)
        
        return self.add_stage(PipelineStage(name="reduce", processor=reduce_processor))
    
    def flatten(self) -> "DataPipeline":
        """Flatten nested lists"""
        def flatten_processor(data: List[List[Any]]) -> List[Any]:
            return [item for sublist in data for item in sublist]
        
        return self.add_stage(PipelineStage(name="flatten", processor=flatten_processor))
    
    def chunk(self, size: int) -> "DataPipeline":
        """Chunk data into smaller pieces"""
        def chunk_processor(data: List[Any]) -> List[List[Any]]:
            return [data[i:i+size] for i in range(0, len(data), size)]
        
        return self.add_stage(PipelineStage(name="chunk", processor=chunk_processor))
