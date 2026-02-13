"""
Tests for PyAgent Orchestrator Module
=====================================

Tests for Task, Workflow, Orchestrator, and AgentPatterns.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestOrchestratorImport:
    """Test that all orchestrator components can be imported."""
    
    def test_import_orchestrator_module(self):
        """Test importing the orchestrator module."""
        from pyagent import orchestrator
        assert orchestrator is not None
    
    def test_import_orchestrator_class(self):
        """Test importing the Orchestrator class."""
        from pyagent.orchestrator import Orchestrator
        assert Orchestrator is not None
    
    def test_import_task_class(self):
        """Test importing the Task class."""
        from pyagent.orchestrator import Task
        assert Task is not None
    
    def test_import_workflow_class(self):
        """Test importing the Workflow class."""
        from pyagent.orchestrator import Workflow
        assert Workflow is not None
    
    def test_import_scheduled_job_class(self):
        """Test importing the ScheduledJob class."""
        from pyagent.orchestrator import ScheduledJob
        assert ScheduledJob is not None
    
    def test_import_task_status_enum(self):
        """Test importing the TaskStatus enum."""
        from pyagent.orchestrator import TaskStatus
        assert TaskStatus is not None
        assert hasattr(TaskStatus, 'PENDING')
        assert hasattr(TaskStatus, 'RUNNING')
        assert hasattr(TaskStatus, 'COMPLETED')
        assert hasattr(TaskStatus, 'FAILED')
    
    def test_import_execution_pattern_enum(self):
        """Test importing the ExecutionPattern enum."""
        from pyagent.orchestrator import ExecutionPattern
        assert ExecutionPattern is not None
        assert hasattr(ExecutionPattern, 'SEQUENTIAL')
        assert hasattr(ExecutionPattern, 'PARALLEL')
        assert hasattr(ExecutionPattern, 'SUPERVISOR')
    
    def test_import_agent_patterns_class(self):
        """Test importing the AgentPatterns class."""
        from pyagent.orchestrator import AgentPatterns
        assert AgentPatterns is not None


class TestTaskStatus:
    """Tests for TaskStatus enum."""
    
    def test_all_status_values(self):
        """Test all TaskStatus values exist."""
        from pyagent.orchestrator import TaskStatus
        
        statuses = ['PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'RETRYING']
        for status in statuses:
            assert hasattr(TaskStatus, status)


class TestExecutionPattern:
    """Tests for ExecutionPattern enum."""
    
    def test_all_pattern_values(self):
        """Test all ExecutionPattern values exist."""
        from pyagent.orchestrator import ExecutionPattern
        
        patterns = ['SEQUENTIAL', 'PARALLEL', 'SUPERVISOR', 'COLLABORATIVE', 
                   'BROADCAST', 'ROUTER', 'CONSENSUS']
        for pattern in patterns:
            assert hasattr(ExecutionPattern, pattern)


class TestTask:
    """Tests for Task dataclass."""
    
    def test_create_task(self):
        """Test creating a task."""
        from pyagent.orchestrator import Task, TaskStatus
        
        task = Task(
            id="test-001",
            name="Test Task"
        )
        
        assert task.id == "test-001"
        assert task.name == "Test Task"
        assert task.status == TaskStatus.PENDING
    
    def test_task_default_status(self):
        """Test that task has default PENDING status."""
        from pyagent.orchestrator import Task, TaskStatus
        
        task = Task(id="test", name="test")
        assert task.status == TaskStatus.PENDING
    
    def test_task_with_kwargs(self):
        """Test creating a task with kwargs."""
        from pyagent.orchestrator import Task
        
        task = Task(
            id="test",
            name="test",
            kwargs={"key": "value"}
        )
        
        assert task.kwargs == {"key": "value"}


class TestWorkflow:
    """Tests for Workflow dataclass."""
    
    def test_create_workflow(self):
        """Test creating a workflow."""
        from pyagent.orchestrator import Workflow, Task
        
        task1 = Task(id="t1", name="Task 1")
        task2 = Task(id="t2", name="Task 2")
        
        workflow = Workflow(
            name="Test Workflow",
            steps=[task1, task2]
        )
        
        assert workflow.name == "Test Workflow"
        assert len(workflow.steps) == 2
    
    def test_workflow_add_step(self):
        """Test adding steps to a workflow."""
        from pyagent.orchestrator import Workflow, Task
        
        task1 = Task(id="t1", name="Task 1")
        task2 = Task(id="t2", name="Task 2")
        
        workflow = Workflow(name="Test Workflow")
        workflow.add_step(task1)
        workflow.add_step(task2)
        
        assert len(workflow.steps) == 2
        assert workflow.steps[0].name == "Task 1"


class TestOrchestrator:
    """Tests for Orchestrator class."""
    
    def test_create_orchestrator(self):
        """Test creating an orchestrator."""
        from pyagent.orchestrator import Orchestrator
        
        orch = Orchestrator()
        assert orch is not None
    
    def test_orchestrator_has_submit_method(self):
        """Test that orchestrator has submit method."""
        from pyagent.orchestrator import Orchestrator
        
        orch = Orchestrator()
        assert hasattr(orch, 'submit')
        assert callable(orch.submit)
    
    def test_orchestrator_has_schedule_method(self):
        """Test that orchestrator has schedule method."""
        from pyagent.orchestrator import Orchestrator
        
        orch = Orchestrator()
        assert hasattr(orch, 'schedule')
        assert callable(orch.schedule)
    
    def test_orchestrator_has_status_method(self):
        """Test that orchestrator has status method."""
        from pyagent.orchestrator import Orchestrator
        
        orch = Orchestrator()
        assert hasattr(orch, 'status')
        assert callable(orch.status)
    
    def test_orchestrator_has_on_method(self):
        """Test that orchestrator has on (event) method."""
        from pyagent.orchestrator import Orchestrator
        
        orch = Orchestrator()
        assert hasattr(orch, 'on')
        assert callable(orch.on)
    
    def test_orchestrator_has_emit_method(self):
        """Test that orchestrator has emit method."""
        from pyagent.orchestrator import Orchestrator
        
        orch = Orchestrator()
        assert hasattr(orch, 'emit')
        assert callable(orch.emit)


class TestAgentPatterns:
    """Tests for AgentPatterns class."""
    
    def test_supervisor_pattern_exists(self):
        """Test that supervisor pattern exists."""
        from pyagent.orchestrator import AgentPatterns
        
        assert hasattr(AgentPatterns, 'supervisor')
        assert callable(AgentPatterns.supervisor)
    
    def test_consensus_pattern_exists(self):
        """Test that consensus pattern exists."""
        from pyagent.orchestrator import AgentPatterns
        
        assert hasattr(AgentPatterns, 'consensus')
        assert callable(AgentPatterns.consensus)
    
    def test_debate_pattern_exists(self):
        """Test that debate pattern exists."""
        from pyagent.orchestrator import AgentPatterns
        
        assert hasattr(AgentPatterns, 'debate')
        assert callable(AgentPatterns.debate)
    
    def test_chain_of_thought_pattern_exists(self):
        """Test that chain_of_thought pattern exists."""
        from pyagent.orchestrator import AgentPatterns
        
        assert hasattr(AgentPatterns, 'chain_of_thought')
        assert callable(AgentPatterns.chain_of_thought)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
